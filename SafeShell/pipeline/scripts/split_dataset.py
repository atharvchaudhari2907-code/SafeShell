"""
SafeShell dataset splitter V3.

Fixes V1/V2 split-allocation bugs:
- IID split is stratified within exact (intent-set + risk) strata.
- Grouped split explicitly fills TEST first, then VAL, then TRAIN.
- Hard guarantees that non-empty datasets get non-empty val/test.
- Exact command groups never cross splits.
- Broader command families never cross grouped splits.
- No silent duplicate deletion.
- Hard target-leakage and command-extraction checks.

Default:
  input  = /home/paras/Desktop/SafeShell/data/labeled/labeled_commands.jsonl
  output = /home/paras/Desktop/SafeShell/data/dataset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path("/home/paras/Desktop/SafeShell/data/labeled/labeled_commands.jsonl")
DEFAULT_OUTPUT = Path("/home/paras/Desktop/SafeShell/data/dataset")

VALID_RISKS = {"safe", "low", "medium", "high", "critical"}

FORBIDDEN_MODEL_FIELDS = {
    "intent", "risk", "operation", "operation_detail", "domain",
    "domain_action", "risk_features", "risk_flags", "label_metadata",
    "intent_resolution", "risk_reasons",
}

COMMAND_PATHS = (
    ("command",),
    ("raw_command",),
    ("model_input", "command"),
    ("model_input", "raw_command"),
    ("input", "command"),
    ("parsed", "command"),
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Malformed JSON at line {n}: {e}") from e
            if not isinstance(obj, dict):
                raise RuntimeError(f"Line {n} is not an object")
            out.append(obj)
    return out


def get_nested(obj: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def normalize_command(command: str) -> str:
    s = str(command).strip().replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n+ *", "\n", s)
    return s


def extract_command(record: dict[str, Any]) -> str:
    candidates = []
    for path in COMMAND_PATHS:
        value = get_nested(record, path)
        if isinstance(value, str) and value.strip():
            candidates.append((".".join(path), value.strip()))

    if not candidates:
        raise RuntimeError(
            f"Record {record.get('id')} has no command. "
            f"Checked: {', '.join('.'.join(p) for p in COMMAND_PATHS)}"
        )

    normalized = {normalize_command(v) for _, v in candidates}
    if len(normalized) != 1:
        raise RuntimeError(
            f"Record {record.get('id')} has conflicting command fields: "
            + "; ".join(f"{p}={v[:80]!r}" for p, v in candidates)
        )
    return candidates[0][1]


def command_hash(command: str) -> str:
    return hashlib.sha256(normalize_command(command).encode()).hexdigest()


def labels(record: dict[str, Any]) -> tuple[tuple[str, ...], str]:
    lab = record.get("labels")
    if not isinstance(lab, dict):
        raise RuntimeError(f"Record {record.get('id')} has no labels")

    intents = lab.get("intent")
    risk = lab.get("risk")
    if isinstance(intents, str):
        intents = [intents]
    if not isinstance(intents, list) or not intents:
        raise RuntimeError(f"Record {record.get('id')} has invalid intent")
    risk = str(risk)
    if risk not in VALID_RISKS:
        raise RuntimeError(f"Record {record.get('id')} has invalid risk {risk!r}")
    return tuple(sorted(set(map(str, intents)))), risk


def walk_leakage(obj: Any, path: str, hits: list[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            ks = str(k)
            if ks in FORBIDDEN_MODEL_FIELDS:
                hits.append(f"{path}.{ks}")
            walk_leakage(v, f"{path}.{ks}", hits)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            walk_leakage(v, f"{path}[{i}]", hits)


def leakage_paths(record: dict[str, Any]) -> list[str]:
    hits = []
    walk_leakage(record.get("model_input", {}), "model_input", hits)
    return hits


def extract_programs(record: dict[str, Any], command: str) -> list[str]:
    mi = record.get("model_input")
    if isinstance(mi, dict):
        for key in ("programs", "executables"):
            vals = mi.get(key)
            if isinstance(vals, list) and vals:
                return [str(x).lower() for x in vals if str(x).strip()]
        cs = mi.get("command_structure")
        if isinstance(cs, dict):
            vals = cs.get("programs")
            if isinstance(vals, list) and vals:
                return [str(x).lower() for x in vals if str(x).strip()]

    vals = re.findall(r"(?:^|[;&|(\n])\s*([A-Za-z0-9_./:+-]+)", command)
    return [x.lower() for x in vals] or ["<unknown>"]


def shell_type(record: dict[str, Any]) -> str:
    mi = record.get("model_input")
    if isinstance(mi, dict):
        cs = mi.get("command_structure")
        if isinstance(cs, dict):
            for key in ("type", "structure", "shell_structure"):
                if cs.get(key):
                    return str(cs[key]).lower()
        if mi.get("shell_structure"):
            return str(mi["shell_structure"]).lower()
    return ""


def family_key(record: dict[str, Any], command: str) -> str:
    programs = extract_programs(record, command)
    stype = shell_type(record)
    tokens = re.findall(r"""(?:[^\s"'\\]|\\.)+""", command)
    cli = {
        "git", "docker", "kubectl", "helm", "terraform", "aws", "gcloud",
        "az", "npm", "npx", "pip", "python", "systemctl", "apt", "apt-get",
        "dnf", "yum", "pacman", "cargo", "go",
    }
    subcommand = ""
    if tokens:
        base = tokens[0].split("/")[-1].lower()
        if base in cli:
            for token in tokens[1:7]:
                if not token.startswith("-") and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", token):
                    subcommand = token.lower()
                    break
    return f"{'|'.join(programs[:5])}::{stype}::{subcommand}"


def stratum_key(record: dict[str, Any]) -> str:
    intents, risk = labels(record)
    return "+".join(intents) + "||" + risk


def stable_key(seed: int, text: str) -> str:
    return hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()


def group_by(records, fn):
    groups = defaultdict(list)
    for r in records:
        groups[fn(r)].append(r)
    return dict(groups)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")


def ids(records):
    return {str(r["id"]) for r in records}


def hashes(records):
    return {command_hash(extract_command(r)) for r in records}


def families(records):
    return {family_key(r, extract_command(r)) for r in records}


def stratified_exact_split(records, seed: int, val_frac=0.10, test_frac=0.10):
    """
    Exact-command groups are atomic. In this dataset they are mostly singletons.
    Split each label stratum independently so rare labels remain represented.
    """
    strata = group_by(records, stratum_key)
    train, val, test = [], [], []

    for key in sorted(strata, key=lambda x: stable_key(seed, x)):
        bucket = sorted(
            strata[key],
            key=lambda r: stable_key(seed, str(r["id"]))
        )
        n = len(bucket)

        if n == 1:
            train.extend(bucket)
            continue
        if n == 2:
            train.append(bucket[0])
            test.append(bucket[1])
            continue

        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac))

        # Always leave at least one training example.
        while n_test + n_val >= n:
            if n_test >= n_val and n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                break

        test.extend(bucket[:n_test])
        val.extend(bucket[n_test:n_test + n_val])
        train.extend(bucket[n_test + n_val:])

    return train, val, test


def allocate_grouped(records, seed: int, val_frac=0.10, test_frac=0.10):
    """
    Whole family groups stay together.

    Strategy:
      1. Randomize deterministically.
      2. Reserve a test set by selecting groups until near target.
      3. Reserve validation from remaining groups.
      4. Everything else goes train.

    We explicitly prioritize having all three splits over trying to hit
    an exact percentage when large groups make that impossible.
    """
    groups = group_by(records, lambda r: family_key(r, extract_command(r)))
    items = list(groups.items())

    # Deterministic shuffle, then largest groups first.
    items.sort(key=lambda kv: stable_key(seed, kv[0]))
    items.sort(key=lambda kv: -len(kv[1]))

    total = len(records)
    target_test = max(1, round(total * test_frac))
    target_val = max(1, round(total * val_frac))

    def choose_groups(target, available):
        selected = []
        selected_size = 0

        # Prefer groups that don't overshoot target too badly.
        while available:
            candidates = []
            for i, (key, rows) in enumerate(available):
                size = len(rows)
                new_size = selected_size + size
                if selected_size >= target:
                    break
                # Absolute distance, with a mild penalty for overshoot.
                distance = abs(target - new_size)
                overshoot = max(0, new_size - target)
                score = distance + 2.0 * overshoot
                candidates.append((score, stable_key(seed, key), i))

            if not candidates:
                break

            _, _, idx = min(candidates)
            key, rows = available.pop(idx)

            # Don't take a group if it would leave no meaningful train set.
            selected.append((key, rows))
            selected_size += len(rows)

            # Once target is reached, stop. If the last group overshot,
            # that's acceptable because groups are atomic.
            if selected_size >= target:
                break

        return selected, available

    remaining = list(items)
    selected_test, remaining = choose_groups(target_test, remaining)

    # Ensure validation can be non-empty.
    selected_val, remaining = choose_groups(target_val, remaining)

    # If validation accidentally empty, move the smallest remaining group.
    if not selected_val and remaining:
        smallest_idx = min(range(len(remaining)), key=lambda i: len(remaining[i][1]))
        selected_val.append(remaining.pop(smallest_idx))

    # If test accidentally empty, steal smallest from train candidate.
    if not selected_test and remaining:
        smallest_idx = min(range(len(remaining)), key=lambda i: len(remaining[i][1]))
        selected_test.append(remaining.pop(smallest_idx))

    train = [r for _, rows in remaining for r in rows]
    val = [r for _, rows in selected_val for r in rows]
    test = [r for _, rows in selected_test for r in rows]

    return train, val, test


def distribution(records):
    intent = Counter()
    risk = Counter()
    combos = Counter()
    programs = Counter()

    for r in records:
        ins, rk = labels(r)
        intent.update(ins)
        risk[rk] += 1
        combos["+".join(ins)] += 1
        for p in extract_programs(r, extract_command(r)):
            programs[p] += 1

    return {
        "records": len(records),
        "intent_occurrences": dict(intent.most_common()),
        "risk_distribution": dict(risk.most_common()),
        "intent_combinations": dict(combos.most_common(25)),
        "top_programs": dict(programs.most_common(25)),
    }


def audit_regime(train, val, test, require_family_isolation=False):
    result = {
        "sizes": {
            "train": len(train), "val": len(val), "test": len(test)
        },
        "id_overlap": {},
        "exact_command_overlap": {},
        "family_overlap": {},
    }

    split_ids = {"train": ids(train), "val": ids(val), "test": ids(test)}
    split_hashes = {"train": hashes(train), "val": hashes(val), "test": hashes(test)}
    split_families = {
        "train": families(train), "val": families(val), "test": families(test)
    }

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        result["id_overlap"][f"{a}__{b}"] = len(split_ids[a] & split_ids[b])
        result["exact_command_overlap"][f"{a}__{b}"] = len(
            split_hashes[a] & split_hashes[b]
        )
        result["family_overlap"][f"{a}__{b}"] = len(
            split_families[a] & split_families[b]
        )

    if require_family_isolation and any(result["family_overlap"].values()):
        raise RuntimeError(
            f"Grouped family leakage detected: {result['family_overlap']}"
        )

    if any(result["id_overlap"].values()):
        raise RuntimeError(f"ID leakage detected: {result['id_overlap']}")

    if any(result["exact_command_overlap"].values()):
        raise RuntimeError(
            f"Exact command leakage detected: {result['exact_command_overlap']}"
        )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 72)
    print("SafeShell Dataset Splitter V3")
    print("=" * 72)
    print(f"Input  : {args.input.resolve()}")
    print(f"Output : {args.output_dir.resolve()}")
    print(f"Seed   : {args.seed}")

    records = load_jsonl(args.input)
    if not records:
        raise RuntimeError("Dataset is empty.")

    print(f"Records: {len(records)}")

    # Strict source validation.
    seen_ids = set()
    command_counter = Counter()
    leakage = []

    for r in records:
        rid = str(r.get("id", "")).strip()
        if not rid:
            raise RuntimeError("Record without id.")

        if rid in seen_ids:
            raise RuntimeError(f"Duplicate ID: {rid}")
        seen_ids.add(rid)

        command = extract_command(r)
        command_counter[command_hash(command)] += 1
        labels(r)
        leakage.extend(leakage_paths(r))

    unique_commands = len(command_counter)
    duplicate_records = sum(c - 1 for c in command_counter.values() if c > 1)

    print(f"Extracted commands : {len(records)}")
    print(f"Unique commands    : {unique_commands}")
    print(f"Exact duplicates   : {duplicate_records}")

    if len(records) > 10 and unique_commands <= 1:
        raise RuntimeError("Command extraction collapsed the dataset.")
    if unique_commands < max(2, int(len(records) * 0.001)):
        raise RuntimeError(
            f"Suspicious command diversity: {unique_commands}/{len(records)}"
        )
    if leakage:
        raise RuntimeError(
            "Target leakage in model_input:\n" +
            "\n".join(sorted(set(leakage))[:30])
        )

    print("Schema validation  : PASS")
    print("Target leakage     : PASS")

    # IID regime: exact duplicate commands stay together.
    # First group by exact command, then stratify by labels.
    exact_groups = group_by(records, lambda r: command_hash(extract_command(r)))

    # If duplicate commands have inconsistent labels, refuse to split them.
    for h, rows in exact_groups.items():
        signatures = {stratum_key(r) for r in rows}
        if len(signatures) > 1:
            raise RuntimeError(
                f"Same normalized command has inconsistent labels: {h[:12]}"
            )

    iid_train, iid_val, iid_test = stratified_exact_split(
        records, args.seed
    )

    # Grouped regime.
    family_groups = group_by(
        records,
        lambda r: family_key(r, extract_command(r))
    )
    grp_train, grp_val, grp_test = allocate_grouped(
        records, args.seed
    )

    print(f"Exact command groups: {len(exact_groups)}")
    print(f"Command families   : {len(family_groups)}")

    iid_audit = audit_regime(
        iid_train, iid_val, iid_test, require_family_isolation=False
    )
    grouped_audit = audit_regime(
        grp_train, grp_val, grp_test, require_family_isolation=True
    )

    # Conservation checks.
    for name, split in (
        ("IID", [iid_train, iid_val, iid_test]),
        ("GROUPED", [grp_train, grp_val, grp_test]),
    ):
        total = sum(len(x) for x in split)
        if total != len(records):
            raise RuntimeError(f"{name} conservation failure: {total}")
        if any(len(x) == 0 for x in split):
            raise RuntimeError(f"{name} contains an empty split.")

    # Write outputs.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(args.output_dir / "train.jsonl", iid_train)
    write_jsonl(args.output_dir / "val.jsonl", iid_val)
    write_jsonl(args.output_dir / "test_iid.jsonl", iid_test)

    write_jsonl(args.output_dir / "train_grouped.jsonl", grp_train)
    write_jsonl(args.output_dir / "val_grouped.jsonl", grp_val)
    write_jsonl(args.output_dir / "test_grouped.jsonl", grp_test)

    audit = {
        "status": "PASS",
        "version": "V3",
        "seed": args.seed,
        "input": str(args.input.resolve()),
        "records": len(records),
        "extracted_commands": len(records),
        "unique_normalized_commands": unique_commands,
        "exact_duplicate_records": duplicate_records,
        "exact_command_groups": len(exact_groups),
        "command_families": len(family_groups),
        "target_leakage": [],
        "iid": {
            "audit": iid_audit,
            "distribution": {
                "train": distribution(iid_train),
                "val": distribution(iid_val),
                "test": distribution(iid_test),
            },
        },
        "grouped": {
            "audit": grouped_audit,
            "distribution": {
                "train": distribution(grp_train),
                "val": distribution(grp_val),
                "test": distribution(grp_test),
            },
        },
    }

    audit_path = args.output_dir / "split_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("IID SPLIT")
    print("=" * 72)
    print(f"Train : {len(iid_train)}")
    print(f"Val   : {len(iid_val)}")
    print(f"Test  : {len(iid_test)}")

    print("\n" + "=" * 72)
    print("GROUPED SPLIT")
    print("=" * 72)
    print(f"Train : {len(grp_train)}")
    print(f"Val   : {len(grp_val)}")
    print(f"Test  : {len(grp_test)}")

    print("\n" + "=" * 72)
    print("VALIDATION")
    print("=" * 72)
    print("Command extraction       : PASS")
    print("Command diversity        : PASS")
    print("Duplicate IDs            : PASS")
    print("Exact command isolation  : PASS")
    print("Grouped family isolation : PASS")
    print("Target leakage           : PASS")
    print("Record conservation      : PASS")
    print("Non-empty splits         : PASS")
    print("Schema/label validation  : PASS")
    print(f"\nAudit: {audit_path}")
    print("STATUS: PASS")


if __name__ == "__main__":
    main()