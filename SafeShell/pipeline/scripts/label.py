"""
SafeShell Labels V15
===================

Converts enriched SafeShell command records into deterministic supervised
learning labels.

Pipeline:
    commands.jsonl
        -> enrich.py V7
        -> ontology.py V2
        -> labels.py V9
        -> labeled_commands.jsonl
        -> dataset split
        -> fast ML model

V5 fixes:
    - Removes the unsafe dataset-wide known-function -> execute fallback.
    - Detects conflicting duplicate commands before deduplication and fails.
    - Deduplicates only exact semantic duplicates (same intent + risk).
    - Keeps unknown ontology rows in review_commands.jsonl.
    - Makes function-definition handling safer: a definition is safe only when
      the command contains no subsequent invocation/side-effecting stage.
    - Uses parsed stage semantics where available for risk decisions.
    - Adds explicit remote/network state-change handling.
    - Strengthens catastrophic/destructive risk detection.
    - Avoids treating read-only network inspection as a state-changing action.
    - Adds semantic audit statistics in addition to structural validation.
    - Keeps model_input free of ontology/risk target fields.
    - Uses the actual V7 enriched dataset by default, with an explicit
      --input override available.
    - Uses atomic output writes.

Run from SafeShell root:
    python3 scripts/labels.py

Optional:
    python3 scripts/labels.py \
        --input data/enriched/enriched_commands_v7.jsonl \
        --output-dir data/labeled
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from ontology import INTENTS, resolve_intent
except ImportError as exc:
    raise SystemExit(
        "Could not import ontology.py. Put labels.py beside ontology.py "
        "inside SafeShell/scripts/."
    ) from exc


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = BASE_DIR / "data" / "enriched" / "enriched_commands_v7.jsonl"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "labeled"

# Backward-compatible fallback for projects that already generated v11.
V11_FALLBACK = BASE_DIR / "data" / "enriched" / "enriched_commands_v11.jsonl"

RISK_LEVELS = ["safe", "low", "medium", "high", "critical"]
LABEL_VERSION = "15.0"

# These fields are produced by enrichment/ontology/risk labeling and must not
# become model features. "labels" and "label_metadata" are also prohibited
# defensively even if they appear only in a future upstream schema.
LEAKAGE_FIELDS = {
    "domain",
    "operation",
    "operation_detail",
    "domain_action",
    "risk_features",
    "security_actions",
    "description",
    "tags",
    "category",
    "subcategory",
    "risk",
    "intent",
    "labels",
    "label_metadata",
    "target_types",
    "risk_flags",
}

# Intents that are intrinsically mutating when assigned by the ontology.
# Broad domain intents such as version_control, manage_container,
# manage_network, manage_database, manage_cloud, and transfer also contain
# read-only operations, so they must not be used as a blanket safe-risk
# conflict rule.
STATE_CHANGING_INTENTS = {
    # These labels are intrinsically mutating in the current ontology.
    # Broad labels such as create/manage_service can also describe read-only
    # or structural behavior, so they are intentionally excluded here.
    "delete",
    "manage_permissions",
    "uninstall",
    "install",
    "move",
    "copy",
}


STRONG_INTENTS = {
    "delete",
    "manage_permissions",
    "uninstall",
    "manage_process",
    "manage_identity",
    "manage_service",
    "manage_container",
    "manage_cloud",
    "manage_database",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _risk_features(record: dict[str, Any]) -> dict[str, Any]:
    enrichment = record.get("enrichment") or {}
    features = enrichment.get("risk_features")
    return features if isinstance(features, dict) else {}


def _bool(data: dict[str, Any], key: str) -> bool:
    return bool(data.get(key, False))


def _command(record: dict[str, Any]) -> str:
    return str(record.get("command") or "").strip()


def _normalized_command(command: str) -> str:
    return re.sub(r"\s+", " ", command.strip())


def _stages(record: dict[str, Any]) -> list[dict[str, Any]]:
    enrichment = record.get("enrichment") or {}
    structure = enrichment.get("command_structure") or {}
    stages = structure.get("stages") or []
    return [s for s in stages if isinstance(s, dict)]


def _stage_text(stage: dict[str, Any]) -> str:
    segment = stage.get("segment")
    if isinstance(segment, str) and segment.strip():
        return segment.strip()

    parts = []
    for key in ("program", "subcommand"):
        value = stage.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())

    flags = stage.get("flags") or []
    args = stage.get("arguments") or []
    if isinstance(flags, list):
        parts.extend(str(x) for x in flags)
    if isinstance(args, list):
        parts.extend(str(x) for x in args)

    return " ".join(parts)


def _has_git_clean_dry_run(command: str) -> bool:
    return bool(
        re.search(
            r"\bgit\s+clean\b[^|;&]*"
            r"(?:--dry-run|\s-n(?:\s|$)|\s-[^- \t]*n[^- \t]*(?:\s|$))",
            command.lower(),
        )
    )


def _contains_redirection(record: dict[str, Any], command: str) -> bool:
    if bool(record.get("has_redirection")):
        return True
    enrichment = record.get("enrichment") or {}
    structure = enrichment.get("command_structure") or {}
    if structure.get("redirections"):
        return True
    return bool(re.search(r"(?:^|\s)(?:>>?|&>>?|<<<|<)\s*", command))


def _is_function_definition_only(record: dict[str, Any]) -> bool:
    """
    A function definition is safe only if it is genuinely definition-only.

    We deliberately do NOT mark:
        function f(){ rm -rf /; }; f
    as safe.
    """
    command = _command(record)
    if not re.match(
        r"^\s*(?:function\s+)?[A-Za-z_][A-Za-z0-9_]*\s*(?:\(\s*\))?\s*\{",
        command,
    ):
        return False

    # A trailing invocation, command separator followed by another command,
    # or multiple parsed stages means the body is not merely being defined.
    if len(_stages(record)) > 1:
        return False

    # If a top-level command separator appears after the closing brace, the
    # definition is followed by another action.
    depth = 0
    closing = -1
    for i, ch in enumerate(command):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
            if depth == 0:
                closing = i
                break

    if closing < 0:
        return False

    tail = command[closing + 1 :].strip().rstrip(";").strip()
    if tail:
        return False

    return True


def _is_read_only_command(record: dict[str, Any]) -> bool:
    command = _command(record)
    c = command.lower()

    if _contains_redirection(record, c):
        return False

    # Network configuration commands are not read-only diagnostics.
    if re.search(r"\bip\s+(?:addr|route|link|neigh)\s+(?:set|add|del|delete|replace|change|append|prepend|remove)\b", c):
        return False

    # A later stage that can write or execute arbitrary code defeats a
    # read-only interpretation of an earlier stage.
    if re.search(
        r"\|\s*(?:tee|cp|mv|rm|sed|awk|perl|python(?:3)?|"
        r"python3?|-|xargs)\b",
        c,
    ):
        return False

    patterns = (
        r"^(?:sudo\s+|doas\s+)?(?:pwd|ls|cat|head|tail|less|more|grep|rg|"
        r"awk|cut|sort|uniq|wc|du|df|free|ps|pgrep|journalctl|dmesg|"
        r"uname|whoami|id|groups|env|printenv|which|whereis|file|stat|"
        r"readlink)\b",
        r"^(?:sudo\s+|doas\s+)?systemctl\s+(?:status|is-active|"
        r"is-enabled|is-failed|show|cat|list-units|list-unit-files|"
        r"list-timers|list-dependencies)\b",
        r"^git\s+(?:status|log|diff|show|merge-base|diff-tree|"
        r"stash\s+(?:list|show)|config\s+(?:--get(?:-all|-regexp)?|"
        r"--list|-l|--show-origin|--show-scope)|worktree\s+list|"
        r"remote\s+(?:show|get-url))\b",
        r"^(?:sudo\s+|doas\s+)?(?:ip\s+(?:addr|route|link|neigh)|"
        r"ss|netstat|lsof|dig|nslookup|host)\b",
    )
    return any(re.search(pattern, c) for pattern in patterns)


def _stage_has_write_semantics(stage: dict[str, Any]) -> bool:
    """
    Prefer parsed stage semantics over raw whole-command regexes.

    This is intentionally conservative: if the enrichment stage says an
    operation is state-changing, we count it as such. If it is clearly a
    read/inspect operation, we don't infer a write merely from a word inside
    an argument.
    """
    operation = str(stage.get("operation") or "").lower()
    detail = str(stage.get("operation_detail") or "").lower()
    program = str(stage.get("program") or "").lower()
    subcommand = str(stage.get("subcommand") or "").lower()

    destructive_ops = {
        "delete",
        "delete_or_modify",
        "terminate_process",
        "change_permissions",
        "change_owner",
        "move",
        "copy",
    }
    state_ops = {
        "package_management",
        "manage_service",
        "container_operation",
        "kubernetes_operation",
        "network_configuration",
        "modify_text",
        "manage_user",
        "manage_group",
        "firewall_management",
        "database_operation",
        "cloud_operation",
        "file_transfer",
    }

    if operation in destructive_ops or operation in state_ops:
        return True

    # Version-control is mixed: read-only commands such as status/log/diff/show
    # must not imply a write. Only known mutating subcommands/details do.
    if operation == "version_control":
        readonly = {"status", "log", "diff", "show", "merge-base", "diff-tree"}
        mutating = {
            "add", "commit", "merge", "rebase", "reset", "restore", "switch",
            "checkout", "push", "pull", "fetch", "cherry-pick", "revert",
        }
        if subcommand in readonly:
            return False
        if subcommand in mutating:
            return True

        def _tokens(value: Any) -> list[str]:
            if isinstance(value, list):
                return [str(x).lower() for x in value]
            if isinstance(value, str):
                return value.lower().split()
            return []

        flags = _tokens(stage.get("flags"))
        args = _tokens(stage.get("arguments"))
        details = _tokens(stage.get("operation_detail"))
        tokens = flags + args + details

        if subcommand == "branch":
            readonly_flags = {
                "-a", "-r", "-v", "-vv", "--verbose", "--merged",
                "--no-merged", "--contains", "--no-contains", "--list",
                "--show-current",
            }
            if any(t in readonly_flags or t.startswith("--format=") for t in tokens):
                return False
            mutation_flags = {
                "-d", "-D", "--delete", "--move", "--copy", "--track",
                "--set-upstream-to", "--set-upstream", "--edit-description",
            }
            if any(t in mutation_flags for t in tokens):
                return True
            return bool(args and any(not t.startswith("-") for t in args))

        if subcommand == "stash":
            if tokens and tokens[0] in {"list", "show"}:
                return False
            if tokens and tokens[0] in {
                "push", "save", "pop", "apply", "drop", "clear", "branch",
            }:
                return True
            return False

        if subcommand == "tag":
            readonly_flags = {
                "-l", "--list", "-n", "--contains", "--merged",
                "--no-merged", "--points-at",
            }
            if any(
                t in readonly_flags
                or t.startswith("--sort=")
                or t.startswith("--format=")
                for t in tokens
            ):
                return False
            if any(t in {"-d", "--delete"} for t in tokens):
                return True
            return bool(args and any(not t.startswith("-") for t in args))

        return False

    state_words = (
        "delete",
        "remove",
        "destroy",
        "terminate",
        "install",
        "uninstall",
        "upgrade",
        "update",
        "restart",
        "stop",
        "start",
        "enable",
        "disable",
        "apply",
        "patch",
        "edit",
        "create",
        "modify",
        "change",
        "publish",
        "push",
        "transfer",
        "move",
        "copy",
        "truncate",
    )
    text = " ".join((detail, program, subcommand))
    return any(
        re.search(rf"\b{re.escape(word)}\b", text)
        for word in state_words
    )


# ---------------------------------------------------------------------------
# Risk labeling
# ---------------------------------------------------------------------------


def _semantic_command_class(record: dict[str, Any]) -> tuple[str, list[str]] | None:
    """Return a high-confidence command-semantic risk override.

    This runs before broad enrichment flags.  The goal is to prevent ontology
    metadata from turning read-only queries into mutations and to distinguish
    state-changing subcommands from read-only parent programs.
    """
    original_command = _command(record)
    c = original_command.lower().strip()
    # Remove a leading privilege wrapper for command-family matching.
    bare = re.sub(r"^(?:sudo|doas)\s+", "", c)

    # Pure package/database/account inspection.
    if re.match(r"^dpkg\s+(?:-l|--list|-s|--status|-L|--listfiles)(?:\s|$)", bare):
        return "safe", ["read_only"]
    if re.match(r"^(?:rpm|pacman)\s+(?:-q|-qi|-ql|-Q|--query)(?:\s|$)", bare):
        return "safe", ["read_only"]
    if re.match(r"^getent\s+(?:passwd|group|shadow|hosts|services|protocols|networks)(?:\s|$)", bare):
        return "safe", ["read_only"]
    if re.match(r"^(?:getfacl|getcap)\s", bare):
        return "safe", ["read_only"]
    if re.match(r"^locate(?:\s|$)", bare):
        return "safe", ["read_only"]

    # Read-only disk/device inspection. sudo does not turn a query into a mutation.
    if re.match(r"^(?:fdisk|parted)\s+-l(?:\s|$)", bare):
        return "safe", ["read_only"]

    # Log/status inspection is read-only, including SSH-related logs.
    if re.match(r"^journalctl(?:\s|$)", bare):
        return "safe", ["read_only"]
    if re.match(
        r"^systemctl\s+(?:status|is-active|is-enabled|is-failed|show|cat|"
        r"list-units|list-unit-files|list-timers|list-dependencies)\b",
        bare,
    ):
        return "safe", ["read_only"]
    if re.match(r"^service\s+\S+\s+status(?:\s|$)", bare):
        return "safe", ["read_only"]

    # Read-only container/orchestration queries.
    if re.match(r"^docker\s+(?:inspect|ps|images|logs|stats|top|version|info)\b", bare):
        return "safe", ["read_only"]
    if re.match(r"^docker\s+(?:network|volume)\s+inspect\b", bare):
        return "safe", ["read_only"]
    if re.match(r"^kubectl\s+(?:get|describe|logs|top|diff|cluster-info|version|config\s+(?:get-contexts|current-context))\b", bare):
        return "safe", ["read_only"]
    if re.match(r"^kubectl\s+rollout\s+status\b", bare):
        return "safe", ["read_only"]

    # Read-only cloud/IaC queries.
    if re.match(r"^aws\s+(?:sts\s+get-caller-identity|configure\s+list|s3\s+ls|ec2\s+(?:describe|list)-|iam\s+(?:list|get)-|logs\s+tail)\b", bare):
        return "safe", ["read_only"]
    if re.match(r"^gcloud\s+(?:auth\s+list|config\s+list|projects\s+list|compute\s+instances\s+(?:list|describe))\b", bare):
        return "safe", ["read_only"]
    if re.match(r"^az\s+(?:account\s+(?:show|list)|group\s+list|vm\s+(?:list|show))\b", bare):
        return "safe", ["read_only"]
    if re.match(r"^terraform\s+(?:validate|plan|state\s+(?:list|show))\b", bare):
        return "safe", ["read_only"]

    # Read-only package manager queries. apt update remains a state change.
    if re.match(r"^(?:apt|apt-get)\s+(?:list|search|show|policy|depends|rdepends)\b", bare):
        return "safe", ["read_only"]

    # mount with no operands displays current mounts; actual mount/umount changes state.
    if bare in {"mount", "mount -l", "mount --all --fake"}:
        return "safe", ["read_only"]
    if re.match(r"^mount\s+", bare) or re.match(r"^umount(?:\s|$)", bare):
        return "medium", ["state_change"]

    # Explicit service mutations.
    if re.match(r"^systemctl\s+(?:start|stop|restart|reload|enable|disable|mask|unmask|daemon-reload|reset-failed|kill)\b", bare) or re.match(r"^service\s+\S+\s+(?:start|stop|restart|reload|enable|disable|mask|unmask|kill)\b", bare):
        return ("high" if c.startswith(("sudo ", "doas ")) else "medium"), ["state_change"] + (["privileged"] if c.startswith(("sudo ", "doas ")) else [])

    # Git branch/reset/clean mutations. `-d` is medium; `-D`/forced delete is high.
    if re.match(r"^git\s+branch\b", bare):
        # Use the original command for -D because `c` has already been
        # lower-cased and therefore cannot distinguish -D from -d.
        if (
            re.search(r"(?:^|\s)-D(?:\s|$)", original_command)
            or re.search(r"(?:^|\s)--force(?:\s|$)", bare)
        ):
            return "high", ["destructive_action", "state_change"]
        if re.search(
            r"(?:^|\s)-d(?:\s|$)|(?:^|\s)--delete(?:\s|$)", bare
        ):
            return "medium", ["destructive_action", "state_change"]
        # A branch name, --track, --move, or --copy creates/modifies refs.
        # Query/list forms such as -a, -r, -v, --merged remain read-only.
        if re.search(
            r"(?:^|\s)(?:--track|--set-upstream-to|--set-upstream|--move|--copy)(?:\s|$)",
            bare,
        ) or re.match(
            r"^git\s+branch\s+(?!-(?:a|r|v|vv|verbose)(?:\s|$))(?!--(?:merged|no-merged|contains|no-contains|list|show-current|format)(?:[=\s]|$))\S+",
            bare,
        ):
            return "medium", ["state_change"]
    if re.match(r"^git\s+reset\s+.*--hard\b", bare):
        return "high", ["destructive_action", "state_change"]

    if re.match(r"^git\s+clean\b", bare):
        if _has_git_clean_dry_run(bare):
            return "safe", ["read_only"]
        if re.search(r"(?:^|\s)(?:-\S*f\S*|--force)(?:\s|$)", bare):
            return "high", ["destructive_action", "state_change"]
        # Standard Git requires force for deletion. Keep non-forced clean
        # conservative but do not claim it actually deleted data.
        return "safe", ["read_only"]

    # Other direct Git mutations remain state-changing even when enrichment
    # does not provide operation/subcommand metadata.
    # `git tag` with no mutation flags is a listing operation and must remain
    # read-only; `git tag NAME` creates a tag and `git tag -d NAME` deletes one.
    if re.match(r"^git\s+tag(?:\s+(-l|--list|-n|--contains|--merged|--no-merged|--points-at|--sort(?:=[^\s]+)?)(?:\s|$)|\s*$)", bare):
        return "safe", ["read_only"]

    git_tag_mutation = re.match(r"^git\s+tag\s+", bare)
    if git_tag_mutation:
        if re.search(r"(?:^|\s)(?:-d|--delete)(?:\s|$)", bare):
            return "medium", ["destructive_action", "state_change"]
        return "medium", ["state_change"]

    # Read-only stash queries must be handled before generic Git mutations.
    if re.match(r"^git\s+stash\s+(?:list|show)\b", bare):
        return "safe", ["read_only"]

    git_mut = re.match(
        r"^git\s+(add|commit|merge|rebase|push|pull|fetch|restore|checkout|"
        r"switch|revert|cherry-pick|stash)\b", bare
    )
    if git_mut:
        sub = git_mut.group(1)
        reasons = ["state_change"]
        if sub in {"revert", "cherry-pick"}:
            reasons.append("destructive_action")
        return "medium", reasons

    # Identity mutation.
    if re.match(r"^(?:gpasswd|usermod|useradd|userdel|groupadd|groupdel|passwd)\b", bare):
        if bare.startswith(("userdel", "groupdel")):
            return "high", ["destructive_action", "state_change"]
        return "medium", ["state_change"]

    # Read-only network diagnostics. Explicit mutators such as `ip link set`,
    # `ip addr add`, and `ip route add/del` must never match this branch.
    if re.match(r"^(?:dig|nslookup|host|ss|netstat|lsof)\b", bare):
        return "safe", ["read_only"]
    if re.match(r"^ip\s+(?:addr|route|link|neigh)\s+(?:set|add|del|delete|replace|change|append|prepend|remove)\b", bare):
        return "medium", ["state_change"]
    if re.match(r"^ip\s+(?:addr|route|link|neigh)(?:\s+(?:show|list|get))?(?:\s|$)", bare):
        return "safe", ["read_only"]

    return None


def _pipeline_semantic_intents(record: dict[str, Any]) -> set[str]:
    """Union high-confidence intents across executable pipeline stages only."""
    e = record.get("enrichment") or {}
    structure = e.get("command_structure") or {}
    programs = [str(x).lower() for x in (structure.get("programs") or []) if str(x).strip()]
    if not programs:
        programs = [str(x).lower() for x in (record.get("commands") or []) if str(x).strip()]

    command = _command(record).lower()
    is_pipeline = bool(record.get("has_pipe")) or len(programs) > 1 or "|" in command
    if not is_pipeline:
        return set()

    found: set[str] = set()
    program_set = set(programs)

    if {"rm", "shred", "wipefs"} & program_set:
        found.add("delete")
    if {"kill", "pkill", "killall"} & program_set:
        found.add("manage_process")
    if {"chmod", "chown", "chgrp", "setfacl", "setcap"} & program_set:
        found.add("manage_permissions")
    if {"systemctl", "service"} & program_set and re.search(
        r"\b(?:systemctl\s+\S+|service\s+\S+\s+\S+)", command
    ):
        if re.search(r"\b(?:start|stop|restart|reload|enable|disable|mask|unmask|kill)\b", command):
            found.add("manage_service")
    if "kubectl" in program_set or "docker" in program_set or "helm" in program_set:
        for stage in (structure.get("stages") or []):
            if not isinstance(stage, dict):
                continue
            p = str(stage.get("program") or "").lower()
            sub = str(stage.get("subcommand") or "").lower()
            if p in {"kubectl", "docker", "helm"} and sub in {
                "apply", "delete", "edit", "patch", "scale", "set", "restart",
                "run", "exec", "rm", "rmi", "build", "connect", "disconnect",
            }:
                found.add("manage_container")
                break

    package_programs = {"apt", "apt-get", "pip", "pip3", "npm", "snap", "flatpak"}
    if package_programs & program_set:
        for stage in (structure.get("stages") or []):
            if not isinstance(stage, dict):
                continue
            p = str(stage.get("program") or "").lower()
            sub = str(stage.get("subcommand") or "").lower()
            if p in package_programs and sub in {
                "install", "remove", "purge", "uninstall", "upgrade", "update",
                "ci", "link", "publish",
            }:
                found.add("manage_package")
                break

    if "git" in program_set:
        git_mutations = {
            "add", "commit", "merge", "rebase", "reset", "restore", "switch",
            "checkout", "push", "pull", "fetch", "stash", "cherry-pick",
            "revert",
        }
        for stage in (structure.get("stages") or []):
            if not isinstance(stage, dict):
                continue
            if str(stage.get("program") or "").lower() == "git":
                sub = str(stage.get("subcommand") or "").lower()
                if sub in git_mutations:
                    found.add("version_control")
                    break
    if {"scp", "sftp", "rsync"} & program_set:
        found.add("transfer")
    if {"curl", "wget"} & program_set:
        found.add("network_request")

    # Read/search stages contribute their own semantic intent.
    if {"find", "locate", "grep", "rg"} & program_set:
        found.add("search")
    if {"cat", "head", "tail", "less", "more", "stat", "journalctl", "getent", "getfacl", "getcap"} & program_set:
        found.add("inspect")

    return found


def assign_risk(record: dict[str, Any]) -> tuple[str, list[str]]:
    """
    Deterministic five-class risk policy.

    Priority:
        critical -> high -> medium -> low -> safe

    Risk is based on enrichment evidence plus parsed command/stage semantics.
    It does not use the ML model or target labels.
    """
    enrichment = record.get("enrichment") or {}
    rf = _risk_features(record)
    command = _command(record)
    c = command.lower()
    stages = _stages(record)

    semantic_override = _semantic_command_class(record)

    # Definition-only shell syntax is safe only when there is no invocation
    # or subsequent action.
    if _is_function_definition_only(record):
        return "safe", ["definition_only"]

    if semantic_override is not None:
        return semantic_override

    destructive = _bool(rf, "destructive")
    modifies_data = _bool(rf, "modifies_data")
    modifies_permissions = _bool(rf, "modifies_permissions")
    privileged = _bool(rf, "privileged")
    force = _bool(rf, "force")
    recursive = _bool(rf, "recursive")
    network = _bool(rf, "network_operation")
    security = _bool(rf, "security_sensitive")
    amplified = _bool(rf, "destructive_amplified_stage")
    privileged_destructive = _bool(rf, "destructive_privileged_stage")

    # Parsed stages are evidence, not blind raw-string matching.
    stage_writes = any(_stage_has_write_semantics(stage) for stage in stages)
    modifies_data = modifies_data or stage_writes

    reasons: list[str] = []

    # -----------------------------------------------------------------------
    # Destructive command families
    # -----------------------------------------------------------------------
    if re.search(
        r"(?:^|[\s;&|])sudo\s+rm\s+[^|;&]*"
        r"(?:-[^|;&\s]*r[^|;&\s]*f|-[^|;&\s]*f[^|;&\s]*r)",
        c,
    ):
        destructive = True
        amplified = True
        privileged_destructive = True

    elif re.search(
        r"(?:^|[\s;&|])rm\s+[^|;&]*"
        r"(?:-[^|;&\s]*r[^|;&\s]*f|-[^|;&\s]*f[^|;&\s]*r)",
        c,
    ):
        destructive = True
        amplified = True

    elif re.search(r"(?:^|[\s;&|])rm(?:\s|$)", c):
        destructive = True

    if re.search(
        r"\b(?:shred|wipefs|userdel|groupdel|kill|pkill|killall)\b",
        c,
    ):
        destructive = True

    if re.search(r"\bmkfs(?:\.[\w+-]+)?\b", c):
        destructive = True
        amplified = True

    if re.search(r"\bdd\b[^|;&]*\bof=/dev/", c):
        destructive = True
        amplified = True

    if re.search(r"\bfind\b[^|;&]*\s-delete\b", c):
        destructive = True

    if re.search(r"\bsed\b[^|;&]*\s(?:-i|--in-place)\b", c):
        modifies_data = True

    if re.search(r"\bgit\s+reset\b.*--hard\b", c):
        destructive = True
        amplified = True

    if re.search(r"\bgit\s+clean\b", c) and not _has_git_clean_dry_run(c):
        destructive = True
        amplified = True

    if re.search(
        r"\bgit\s+push\b.*(?:--force|--force-with-lease|\s-f(?:\s|$))",
        c,
    ):
        destructive = True
        amplified = True

    if re.search(
        r"\bgit\s+branch\b.*(?:-D|--delete.*--force)",
        c,
    ):
        destructive = True
        amplified = True

    if re.search(
        r"\b(?:docker|kubectl|helm)\s+"
        r"(?:container\s+)?(?:rm|rmi|delete|destroy|uninstall|prune)\b",
        c,
    ):
        destructive = True
        amplified = True

    if re.search(r"\bkubectl\s+delete\b", c):
        destructive = True
        amplified = True

    if re.search(r"\bterraform\s+destroy\b", c):
        destructive = True
        amplified = True

    if re.search(
        r"\baws\s+ec2\s+terminate-instances\b|"
        r"\bgcloud\s+compute\s+instances\s+delete\b|"
        r"\baz\s+vm\s+delete\b",
        c,
    ):
        destructive = True
        amplified = True

    # -----------------------------------------------------------------------
    # Explicit state-changing families
    # -----------------------------------------------------------------------
    state_patterns = (
        r"\b(?:touch|mkdir|cp|mv|ln|tee|truncate|install)\b",
        r"\b(?:systemctl|service)\s+(?:start|stop|restart|reload|enable|"
        r"disable|mask|unmask|daemon-reload|reset-failed|kill)\b",
        r"\b(?:apt|apt-get|npm|pip3?|snap|flatpak)\s+(?:install|uninstall|"
        r"remove|purge|upgrade|update|ci|link|publish)\b",
        r"\bdocker\s+compose\s+(?:up|down|restart|build|run|pull|push|rm|"
        r"stop|start|create)\b",
        r"\b(?:docker|kubectl|helm)\s+(?:run|exec|build|create|apply|edit|"
        r"patch|scale|set|restart|start|stop|update|upgrade|install|publish|"
        r"push|pull|connect|disconnect|network|volume)\b",
        r"\bterraform\s+(?:apply|import|taint|untaint)\b",
        r"\b(?:aws\s+s3\s+sync|aws\s+ec2\s+(?:start-instances|stop-instances|"
        r"reboot-instances)|gcloud\s+compute\s+instances\s+(?:start|stop|"
        r"reset)|az\s+vm\s+(?:start|stop|restart))\b",
        r"\b(?:chmod|chown|chgrp|setfacl|setcap|usermod|useradd|groupadd|"
        r"passwd)\b",
    )

    if any(re.search(pattern, c) for pattern in state_patterns):
        modifies_data = True

    # -----------------------------------------------------------------------
    # Network semantics
    # -----------------------------------------------------------------------
    remote_execution = bool(
        re.search(
            r"\b(?:ssh|scp|sftp|rsync)\b",
            c,
        )
    )
    remote_write = bool(
        re.search(
            r"\b(?:scp|sftp|rsync)\b",
            c,
        )
    )
    network_delete = bool(
        re.search(
            r"\b(?:curl|wget)\b[^|;&]*"
            r"(?:-x\s*(?:delete|put|patch)|--request\s+"
            r"(?:delete|put|patch))\b",
            c,
        )
    )

    if remote_execution:
        reasons.append("remote_execution")
        network = True

    if remote_write:
        reasons.append("remote_write")
        modifies_data = True

    if network_delete:
        reasons.append("remote_state_change")
        destructive = True
        amplified = True
        network = True

    # Remote HTTP mutation is state-changing even when no other enrichment
    # feature marks the request as a write.

    # -----------------------------------------------------------------------
    # Read-only exceptions
    # -----------------------------------------------------------------------
    if (
        _has_git_clean_dry_run(c)
        or re.search(
            r"\b(?:python3?|pip3?)\s+(?:-m\s+)?pip\s+install\b.*--dry-run\b",
            c,
        )
    ):
        destructive = False
        modifies_data = False
        amplified = False
        privileged_destructive = False

    if (
        _is_read_only_command(record)
        and not destructive
        and not modifies_permissions
        and not remote_execution
        and not remote_write
        and not network_delete
    ):
        reasons.append("read_only")
        if network:
            reasons.append("network_operation")
            return "low" if not privileged else "medium", reasons
        return "safe", reasons

    # -----------------------------------------------------------------------
    # Reasons
    # -----------------------------------------------------------------------
    if destructive:
        reasons.append("destructive_action")
    if modifies_data:
        reasons.append("state_change")
    if modifies_permissions:
        reasons.append("modifies_permissions")
    if privileged:
        reasons.append("privileged")
    if force:
        reasons.append("force")
    if recursive:
        reasons.append("recursive")
    if network:
        reasons.append("network_operation")
    if security:
        reasons.append("security_sensitive")

    # -----------------------------------------------------------------------
    # Impact classification
    # -----------------------------------------------------------------------
    high_destructive = bool(
        re.search(
            r"(?:^|[\s;&|])rm\s+[^|;&]*-[^|;&\s]*r",
            c,
        )
        or re.search(
            r"\b(?:git\s+clean|git\s+reset\s+--hard|"
            r"git\s+push\b.*(?:--force|--force-with-lease)|"
            r"terraform\s+destroy|kubectl\s+delete|"
            r"docker\s+(?:container\s+)?(?:rm|rmi|prune)|"
            r"aws\s+ec2\s+terminate-instances|"
            r"gcloud\s+compute\s+instances\s+delete|"
            r"az\s+vm\s+delete)\b",
            c,
        )
        or re.search(r"\bmkfs(?:\.[\w+-]+)?\b", c)
        or re.search(r"\b(?:userdel|groupdel)\b", c)
        or re.search(r"\bdd\b[^|;&]*\bof=/dev/", c)
        or remote_write and destructive
    )

    # Catastrophic target: broad root/system scope, raw block devices, or
    # explicit privileged recursive deletion.
    catastrophic_target = bool(
        re.search(
            r"\bsudo\s+rm\b[^|;&]*"
            r"(?:-[A-Za-z]*r[A-Za-z]*f|-[A-Za-z]*f[A-Za-z]*r)\b"
            r"[^|;&]*(?:^|\s)/(?:$|\s|etc(?:/|$)|var(?:/|$)|home(?:/|$)|"
            r"usr(?:/|$)|root(?:/|$)|boot(?:/|$)|bin(?:/|$)|sbin(?:/|$)|"
            r"opt(?:/|$)|srv(?:/|$))",
            c,
        )
        or re.search(r"\brm\s+-rf\s+/(?:\*|\s|$)", c)
        or re.search(
            r"\b(?:dd\b[^|;&]*\bof=/dev/|"
            r"mkfs(?:\.[\w+-]+)?\s+/dev/)",
            c,
        )
        or re.search(r"\bfind\s+/\s+[^|;&]*-delete\b", c)
    )

    # Critical is reserved for high-impact destructive actions with a
    # catastrophic target/scope. It is intentionally narrow.
    if catastrophic_target and destructive and (
        privileged_destructive or privileged or amplified
    ):
        reasons.append("catastrophic_scope")
        return "critical", list(dict.fromkeys(reasons))

    if destructive and privileged:
        return "high", list(dict.fromkeys(reasons))

    if destructive and high_destructive:
        return "high", list(dict.fromkeys(reasons))

    if destructive:
        return "medium", list(dict.fromkeys(reasons))

    if modifies_permissions:
        return "medium", list(dict.fromkeys(reasons))

    if modifies_data:
        return "medium", list(dict.fromkeys(reasons))

    # Remote execution is materially different from ordinary network access.
    if remote_execution:
        return "medium" if not privileged else "high", list(dict.fromkeys(reasons))

    if security or (privileged and not _is_read_only_command(record)):
        return "low", list(dict.fromkeys(reasons))

    if network:
        return "low", list(dict.fromkeys(reasons))

    return "safe", list(dict.fromkeys(reasons))


# ---------------------------------------------------------------------------
# Model-input construction
# ---------------------------------------------------------------------------

def build_model_input(record: dict[str, Any]) -> dict[str, Any]:
    """
    Keep inference-available structural features while excluding all target
    and ontology/risk-derived fields.

    The current baseline model may use only command text; these structured
    fields are preserved for later leakage-safe experiments.
    """
    enrichment = record.get("enrichment") or {}
    structure = enrichment.get("command_structure") or {}
    stages = structure.get("stages") or []

    model_input: dict[str, Any] = {
        "command": record.get("command", ""),
        "program": enrichment.get("program"),
        "raw_program": enrichment.get("raw_program"),
        "program_type": enrichment.get("program_type"),
        "subcommand": enrichment.get("subcommand"),
        "program_sequence": [
            stage.get("program")
            for stage in stages
            if isinstance(stage, dict)
        ],
        "subcommand_sequence": [
            stage.get("subcommand")
            for stage in stages
            if isinstance(stage, dict) and stage.get("subcommand")
        ],
        "stage_count": structure.get("pipeline_length", 1),
        "wrappers": enrichment.get("wrappers", []),
        "wrapper_arguments": enrichment.get("wrapper_arguments", []),

        # Original parser structure.
        "commands": record.get("commands", []),
        "flags": record.get("flags", []),
        "arguments": record.get("arguments", []),
        "operators": record.get("operators", []),
        "redirections": record.get("redirections", []),
        "paths": record.get("paths", []),
        "environment_variables": record.get("environment_variables", []),

        # Shell-structure booleans.
        "has_sudo": bool(record.get("has_sudo", False)),
        "has_pipe": bool(record.get("has_pipe", False)),
        "has_redirection": bool(record.get("has_redirection", False)),
        "has_chaining": bool(record.get("has_chaining", False)),
        "has_command_substitution": bool(
            record.get("has_command_substitution", False)
        ),
        "has_variable_assignment": bool(
            record.get("has_variable_assignment", False)
        ),
        "has_glob": bool(record.get("has_glob", False)),
        "has_shell_expansion": bool(
            record.get("has_shell_expansion", False)
        ),
        "has_quotes": bool(record.get("has_quotes", False)),
        "has_subshell": bool(record.get("has_subshell", False)),
        "execution_mode": record.get("execution_mode"),

        # Argument roles and structural semantics are available at inference
        # time and are not themselves targets.
        "argument_roles": enrichment.get("argument_roles", []),
        "command_structure": {
            "type": structure.get("type"),
            "pipeline_length": structure.get("pipeline_length", 1),
            "programs": list(structure.get("programs") or []),
            "operators": list(structure.get("operators") or []),
            "redirections": list(structure.get("redirections") or []),
            "stages": [
                {
                    "program": stage.get("program"),
                    "subcommand": stage.get("subcommand"),
                    "flags": list(stage.get("flags") or []),
                }
                for stage in stages
                if isinstance(stage, dict)
            ],
        },
        "shell_features": dict(enrichment.get("shell_features") or {}),
    }

    return model_input


# ---------------------------------------------------------------------------
# Record labeling
# ---------------------------------------------------------------------------

def _command_intent_overrides(record: dict[str, Any]) -> set[str]:
    """High-confidence command-semantic intent corrections.

    Ontology V2 intentionally maps some broad Git/filesystem operations to
    inspection. For commands whose syntax unambiguously performs a mutation,
    the command itself is stronger evidence than the coarse operation label.
    """
    c = _command(record).lower().strip()
    stages = _stages(record)
    is_pipeline = bool(record.get("has_pipe")) or "|" in c or len(stages) > 1
    if is_pipeline:
        return set()

    overrides: set[str] = set()

    # Git branch: listing/query forms are inspect; creating/deleting/tracking
    # a branch is version control.
    if re.match(r"^git\s+branch(?:\s|$)", c):
        if re.search(
            r"(?:^|\s)(?:-d|-D|--delete|--move|--copy|--set-upstream-to|--track|--set-upstream)(?:\s|$)",
            c,
        ):
            overrides.add("version_control")
        elif re.match(r"^git\s+branch\s+(?!-(?:a|r|v|vv|verbose|merged|no-merged|contains|no-contains|list|show-current|format)(?:\s|$))\S+", c):
            overrides.add("version_control")

    # Git tag: bare/list/query forms are inspect; creation/deletion is version
    # control.
    if re.match(r"^git\s+tag(?:\s|$)", c):
        if re.match(r"^git\s+tag\s*$", c) or re.search(
            r"^git\s+tag\s+(?:-l|--list|--contains|--merged|--no-merged|--points-at|--sort=)",
            c,
        ):
            pass
        else:
            overrides.add("version_control")

    # Filesystem formatting changes filesystem state; the coarse V7 operation
    # `format` can otherwise land in a text-transformation intent.
    if re.match(r"^(?:sudo\s+|doas\s+)?mkfs(?:\.[\w+-]+)?(?:\s|$)", c):
        overrides.add("configure")

    return overrides


def label_record(record: dict[str, Any]) -> dict[str, Any]:
    intent_values: list[str] = []
    sources: list[str] = []
    stages = _stages(record)

    # Resolve every executable stage so pipelines do not suffer from
    # head-program bias.
    if stages:
        for stage in stages:
            stage_record = {
                "id": record.get("id"),
                "command": stage.get("segment") or stage.get("program", ""),
                "enrichment": {
                    "program": stage.get("program"),
                    "domain": stage.get("domain"),
                    "operation": stage.get("operation"),
                    "operation_detail": stage.get("operation_detail"),
                },
            }
            stage_labels, source = resolve_intent(stage_record)
            intent_values.extend(stage_labels)
            sources.append(source)

    if not intent_values:
        direct_labels, source = resolve_intent(record)
        intent_values.extend(direct_labels)
        sources.append(source)

    # Whole-command resolution adds shell-level intent and command overrides.
    direct_labels, direct_source = resolve_intent(record)

    # High-confidence union of executable-stage semantics. This specifically
    # prevents head-program bias such as find | xargs rm -> search only.
    pipeline_intents = _pipeline_semantic_intents(record)
    intent_values.extend(pipeline_intents)
    if pipeline_intents:
        sources.append("pipeline_semantics")

    is_pipeline = (
        bool(record.get("has_pipe"))
        or "|" in _command(record)
        or len(stages) > 1
    )

    if direct_source == "command_override" and not is_pipeline and len(stages) <= 1:
        intent_values = direct_labels
        sources = [direct_source]
    else:
        intent_values.extend(direct_labels)
        sources.append(direct_source)

    command_overrides = _command_intent_overrides(record)
    if command_overrides:
        # Apply after ontology resolution so a high-confidence syntax-level
        # correction cannot be overwritten by a coarse operation mapping.
        intent_values = list(command_overrides)
        sources = list(dict.fromkeys(sources + ["command_semantics"]))

    shell_features = (record.get("enrichment") or {}).get("shell_features") or {}
    shell_control_syntax = bool(
        re.match(
            r"^\s*(?:if|case|for|while|until|function)\b",
            _command(record),
        )
    )

    if (
        any(
            shell_features.get(key)
            for key in (
                "has_loop",
                "has_case",
                "has_condition_test",
                "has_function_definition",
                "has_subshell",
            )
        )
        or shell_control_syntax
    ) and "shell_control" not in intent_values:
        intent_values.append("shell_control")
        sources.append("shell_structure")

    order = {name: index for index, name in enumerate(INTENTS)}
    intents = sorted(
        set(label for label in intent_values if label in INTENTS),
        key=lambda label: order[label],
    )

    # No dataset-wide "known function" fallback. Unknown is safer than
    # inventing an execute label.
    if not intents:
        intents = ["unknown"]
        sources = ["unknown"]

    risk, risk_reasons = assign_risk(record)

    return {
        "id": record.get("id"),
        "model_input": build_model_input(record),
        "labels": {
            "intent": intents,
            "risk": risk,
        },
        "label_metadata": {
            "label_version": LABEL_VERSION,
            "intent_resolution": "+".join(dict.fromkeys(sources)),
            "risk_reasons": risk_reasons,
        },
    }


# ---------------------------------------------------------------------------
# Validation / semantic audit
# ---------------------------------------------------------------------------

def validate_source_record(
    record: dict[str, Any],
) -> tuple[bool, str]:
    if not isinstance(record, dict):
        return False, "record_not_object"
    if "id" not in record:
        return False, "missing_id"
    if "command" not in record:
        return False, "missing_command"
    if "enrichment" not in record or not isinstance(
        record["enrichment"], dict
    ):
        return False, "missing_enrichment"
    return True, ""


def _find_duplicate_conflicts(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """
    Group normalized commands before deduplication.

    A command may have repeated records only when their target labels agree.
    Any conflicting target is a hard dataset error.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        command = _normalized_command(
            str((row.get("model_input") or {}).get("command", ""))
        )
        groups[command].append(row)

    conflicts: dict[str, list[dict[str, Any]]] = {}
    duplicate_counts: dict[str, int] = {}

    for command, items in groups.items():
        if len(items) <= 1:
            continue

        duplicate_counts[command] = len(items)

        semantic_targets = {
            (
                tuple(sorted((item.get("labels") or {}).get("intent", []))),
                (item.get("labels") or {}).get("risk"),
            )
            for item in items
        }

        if len(semantic_targets) > 1:
            conflicts[command] = items

    return conflicts, duplicate_counts


def _deduplicate_semantically_identical(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    seen: dict[str, tuple[tuple[str, ...], str]] = {}
    deduped: list[dict[str, Any]] = []
    duplicate_counts: Counter[str] = Counter()

    for row in rows:
        command = _normalized_command(
            str((row.get("model_input") or {}).get("command", ""))
        )
        labels = row.get("labels") or {}
        target = (
            tuple(sorted(labels.get("intent") or [])),
            str(labels.get("risk")),
        )

        if command in seen:
            duplicate_counts[command] += 1
            # Conflict detection occurs before this function. Therefore a
            # duplicate here is semantically identical and safe to collapse.
            continue

        seen[command] = target
        deduped.append(row)

    return deduped, dict(duplicate_counts)


def audit_labeled_records(
    labeled: list[dict[str, Any]],
    review: list[dict[str, Any]],
    malformed: list[dict[str, Any]],
    duplicate_conflicts: dict[str, list[dict[str, Any]]],
    duplicate_commands: dict[str, int],
) -> dict[str, Any]:

    all_rows = labeled + review

    ids = [row["id"] for row in all_rows]
    duplicate_ids = [
        key for key, value in Counter(ids).items() if value > 1
    ]

    invalid_intents = Counter()
    invalid_risks = Counter()
    intent_counts = Counter()
    risk_counts = Counter()
    source_counts = Counter()
    risk_reason_counts = Counter()

    for row in labeled:
        labels = row["labels"]
        intents = labels["intent"]
        risk = labels["risk"]

        for intent in intents:
            intent_counts[intent] += 1
            if intent not in INTENTS:
                invalid_intents[intent] += 1

        risk_counts[risk] += 1
        if risk not in RISK_LEVELS:
            invalid_risks[risk] += 1

        source_counts[row["label_metadata"]["intent_resolution"]] += 1
        for reason in row["label_metadata"]["risk_reasons"]:
            risk_reason_counts[reason] += 1

    leakage_hits: list[tuple[Any, str]] = []

    def scan(value: Any, row_id: Any, path: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in LEAKAGE_FIELDS:
                    leakage_hits.append((row_id, path + key))
                scan(child, row_id, path + key + ".")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                scan(child, row_id, path + str(index) + ".")

    for row in all_rows:
        scan(row.get("model_input", {}), row.get("id"))

    empty_intent_ids = [
        row["id"]
        for row in labeled
        if not row["labels"].get("intent")
    ]

    intent_risk_conflicts = Counter()

    for row in labeled:
        intents = set(row["labels"].get("intent", []))
        risk = row["labels"].get("risk")

        if risk == "safe" and intents & STATE_CHANGING_INTENTS:
            intent_risk_conflicts["state_changing_intent_marked_safe"] += 1

        if risk == "critical" and not intents & STRONG_INTENTS:
            intent_risk_conflicts[
                "critical_without_strong_state_change_intent"
            ] += 1

        if risk == "safe" and row["label_metadata"].get(
            "risk_reasons"
        ) == ["privileged"]:
            intent_risk_conflicts[
                "privileged_safe_without_read_only_reason"
            ] += 1

    invalid_review_ids = [
        row["id"]
        for row in review
        if (row.get("labels") or {}).get("intent") != ["unknown"]
    ]

    # Semantic audit counters. These do not silently rewrite labels; they
    # surface areas that need review.
    semantic_flags = Counter()

    for row in labeled:
        command = _command({
            "command": (row.get("model_input") or {}).get("command", "")
        })
        labels = row["labels"]
        risk = labels["risk"]
        c = command.lower()

        if re.search(r"\brm\s+-rf\b", c) and risk == "safe":
            semantic_flags["rm_rf_marked_safe"] += 1

        if re.search(r"\bsudo\s+rm\b", c) and risk == "safe":
            semantic_flags["sudo_rm_marked_safe"] += 1

        if re.search(r"\bkubectl\s+delete\b", c) and risk == "safe":
            semantic_flags["kubectl_delete_marked_safe"] += 1

        if re.search(r"\bterraform\s+destroy\b", c) and risk == "safe":
            semantic_flags["terraform_destroy_marked_safe"] += 1

        if re.search(r"\b(?:chmod|chown|setfacl|setcap)\b", c):
            if risk == "safe":
                semantic_flags["permission_change_marked_safe"] += 1

        # Only flag actual SSH client invocations. Service names such as
        # `sshd`, `journalctl -u ssh`, and `systemctl status ssh` are local
        # inspection and must not trigger this audit rule.
        if re.search(r"(?:^|[;&|]\s*)(?:sudo\s+|doas\s+)?ssh(?:\s|$)", c) and risk == "safe":
            semantic_flags["ssh_remote_marked_safe"] += 1

        if re.match(r"^\s*(?:sudo\s+|doas\s+)?(?:dpkg\s+(?:-l|--list|-s|--status|-L|--listfiles)|getent\s+|getfacl\s+|getcap\s+|journalctl(?:\s|$)|systemctl\s+(?:status|is-active|is-enabled|is-failed|show)|mount(?:\s+-l)?\s*$)", c) and risk in {"medium", "high", "critical"}:
            semantic_flags["read_only_marked_risky"] += 1

        if re.search(r"\b(?:systemctl|service)\s+(?:start|stop|restart|reload|enable|disable)\b", c) and risk == "safe":
            semantic_flags["service_mutation_marked_safe"] += 1

        if re.search(r"\bgit\s+branch\s+.*(?:-[dD]|--delete\s+--force)\b", command, re.IGNORECASE) and risk == "safe":
            semantic_flags["git_destructive_branch_marked_safe"] += 1

        if re.search(r"\bgit\s+clean\b", c) and not _has_git_clean_dry_run(c):
            if re.search(r"(?:^|\s)(?:-\S*f\S*|--force)(?:\s|$)", c) and risk != "high":
                semantic_flags["git_force_clean_not_high"] += 1

        if re.match(r"^\s*git\s+tag(?:\s+(?:-l|--list|-n|--contains|--merged|--no-merged|--points-at|--sort(?:=[^\s]+)?))?\s*$", c) and risk != "safe":
            semantic_flags["git_tag_list_marked_risky"] += 1

        if re.search(r"\bfind\b[^|;&]*\|\s*xargs\s+.*\brm\b", c) and "delete" not in labels["intent"]:
            semantic_flags["pipeline_delete_intent_missing"] += 1

        # Only inspect standalone/read-only Git commands. A compound command
        # containing `git status` may legitimately be risky because another
        # stage (e.g. git pull, tee, or package installation) changes state.
        standalone_git_read = re.match(
            r"^\s*(?:sudo\s+|doas\s+)?git\s+(?:status|log|diff|show|remote\s+(?:-v|show|get-url))\s*$",
            c,
        )
        if standalone_git_read and risk in {"medium", "high", "critical"}:
            semantic_flags["git_read_only_marked_risky"] += 1

    # -------------------------------------------------------------------
    # Final semantic invariants. These are intentionally conservative and
    # inspect the actual labeled command, not the upstream target metadata.
    # -------------------------------------------------------------------
    dangerous_safe_patterns = (
        r"\bsudo\s+rm\b",
        r"\brm\s+-[A-Za-z]*r[A-Za-z]*f\b",
        r"\bmkfs(?:\.[\w+-]+)?\s+/dev/",
        r"\bdd\b[^|;&]*\bof=/dev/",
        r"\bkubectl\s+delete\b",
        r"\bterraform\s+destroy\b",
        r"\buserdel\b",
        r"\bgroupdel\b",
    )
    for row in labeled:
        command = str((row.get("model_input") or {}).get("command", "")).lower()
        risk = (row.get("labels") or {}).get("risk")
        if risk == "safe" and any(re.search(p, command) for p in dangerous_safe_patterns):
            semantic_flags["dangerous_pattern_marked_safe"] += 1

    status = "PASS"

    if (
        malformed
        or duplicate_ids
        or duplicate_conflicts
        or invalid_intents
        or invalid_risks
        or empty_intent_ids
        or invalid_review_ids
        or leakage_hits
        or any(semantic_flags.values())
    ):
        status = "FAIL"

    return {
        "label_version": LABEL_VERSION,
        "records_total": len(all_rows),
        "training_records": len(labeled),
        "review_records": len(review),
        "malformed_records": len(malformed),
        "duplicate_ids": duplicate_ids,
        "duplicate_commands": duplicate_commands,
        "duplicate_command_conflicts": {
            command: [
                {
                    "id": row.get("id"),
                    "labels": row.get("labels"),
                }
                for row in rows
            ]
            for command, rows in duplicate_conflicts.items()
        },
        "invalid_intents": dict(invalid_intents),
        "invalid_risks": dict(invalid_risks),
        "empty_intent_ids": empty_intent_ids,
        "invalid_review_ids": invalid_review_ids,
        "intent_risk_conflicts": dict(intent_risk_conflicts),
        "model_input_leakage": leakage_hits,
        "semantic_flags": dict(semantic_flags),
        "intent_distribution": dict(intent_counts),
        "risk_distribution": dict(risk_counts),
        "intent_resolution_sources": dict(source_counts),
        "risk_reason_distribution": dict(risk_reason_counts),
        "status": status,
    }


# ---------------------------------------------------------------------------
# Atomic writers
# ---------------------------------------------------------------------------

def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def atomic_write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    text = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    )
    atomic_write_text(path, text)


def write_text_audit(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "=" * 72,
        "SafeShell Label Audit V14",
        "=" * 72,
        f"Label version       : {audit['label_version']}",
        f"Total records       : {audit['records_total']}",
        f"Training records    : {audit['training_records']}",
        f"Review records      : {audit['review_records']}",
        f"Malformed            : {audit['malformed_records']}",
        f"Duplicate IDs        : {len(audit['duplicate_ids'])}",
        f"Duplicate commands   : {len(audit['duplicate_commands'])}",
        f"Duplicate conflicts  : {len(audit['duplicate_command_conflicts'])}",
        f"Invalid intents      : {sum(audit['invalid_intents'].values())}",
        f"Invalid risks        : {sum(audit['invalid_risks'].values())}",
        "Intent mode          : MULTI-LABEL",
        "",
        "Intent distribution:",
    ]

    for label, count in sorted(
        audit["intent_distribution"].items(),
        key=lambda item: (-item[1], item[0]),
    ):
        lines.append(f"  {label:28} {count}")

    lines += ["", "Risk distribution:"]

    for risk in RISK_LEVELS:
        lines.append(
            f"  {risk:28} {audit['risk_distribution'].get(risk, 0)}"
        )

    lines += ["", "Intent resolution sources:"]

    for source, count in sorted(
        audit["intent_resolution_sources"].items(),
        key=lambda item: (-item[1], item[0]),
    ):
        lines.append(f"  {source:28} {count}")

    lines += ["", "Risk feature reasons:"]

    for reason, count in sorted(
        audit["risk_reason_distribution"].items(),
        key=lambda item: (-item[1], item[0]),
    ):
        lines.append(f"  {reason:28} {count}")

    lines += ["", "Semantic audit flags:"]

    if audit["semantic_flags"]:
        for flag, count in sorted(
            audit["semantic_flags"].items(),
            key=lambda item: (-item[1], item[0]),
        ):
            lines.append(f"  {flag:40} {count}")
    else:
        lines.append("  none")

    lines += [
        "",
        "FINAL STATUS: " + audit["status"],
        "=" * 72,
    ]

    atomic_write_text(path, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(input_path: Path, output_dir: Path) -> dict[str, Any]:
    source_records: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    seen_ids: set[Any] = set()

    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            raw = line.strip()
            if not raw:
                continue

            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                malformed.append(
                    {
                        "line": line_number,
                        "error": str(exc),
                    }
                )
                continue

            valid, reason = validate_source_record(record)

            if not valid:
                malformed.append(
                    {
                        "line": line_number,
                        "id": (
                            record.get("id")
                            if isinstance(record, dict)
                            else None
                        ),
                        "error": reason,
                    }
                )
                continue

            if record["id"] in seen_ids:
                # Duplicate IDs are a structural error, not something to
                # silently discard.
                malformed.append(
                    {
                        "line": line_number,
                        "id": record["id"],
                        "error": "duplicate_id",
                    }
                )
                continue

            seen_ids.add(record["id"])
            source_records.append(record)

    # Label every source record. Unknown intent is deliberately retained in
    # review rather than fabricated into execute.
    labeled_candidates: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []

    for record in source_records:
        labeled_record = label_record(record)

        if labeled_record["labels"]["intent"] == ["unknown"]:
            review.append(labeled_record)
        else:
            labeled_candidates.append(labeled_record)

    # Detect conflicting duplicate commands BEFORE deduplication.
    duplicate_conflicts, duplicate_commands = _find_duplicate_conflicts(
        labeled_candidates
    )

    # A semantic conflict is not safe to train on. Fail rather than selecting
    # an arbitrary first occurrence.
    if duplicate_conflicts:
        output_dir.mkdir(parents=True, exist_ok=True)

        conflict_audit = {
            "label_version": LABEL_VERSION,
            "status": "FAIL",
            "reason": "conflicting_duplicate_command_labels",
            "duplicate_command_conflicts": {
                command: [
                    {
                        "id": row.get("id"),
                        "labels": row.get("labels"),
                    }
                    for row in rows
                ]
                for command, rows in duplicate_conflicts.items()
            },
        }

        atomic_write_text(
            output_dir / "label_audit_v15.json",
            json.dumps(
                conflict_audit,
                indent=2,
                ensure_ascii=False,
            ),
        )

        raise SystemExit(
            "LABEL AUDIT FAILED: conflicting labels found for normalized "
            "duplicate commands. See data/labeled/label_audit_v15.json."
        )

    labeled, removed_duplicates = _deduplicate_semantically_identical(
        labeled_candidates
    )

    labeled.sort(
        key=lambda row: (str(type(row["id"])), row["id"])
    )
    review.sort(
        key=lambda row: (str(type(row["id"])), row["id"])
    )

    audit = audit_labeled_records(
        labeled=labeled,
        review=review,
        malformed=malformed,
        duplicate_conflicts=duplicate_conflicts,
        duplicate_commands=duplicate_commands,
    )

    audit["source_records"] = len(source_records)
    audit["deduplicated_training_records"] = len(labeled)
    audit["removed_semantic_duplicates"] = removed_duplicates

    output_dir.mkdir(parents=True, exist_ok=True)

    atomic_write_jsonl(
        output_dir / "labeled_commands.jsonl",
        labeled,
    )
    atomic_write_jsonl(
        output_dir / "review_commands.jsonl",
        review,
    )

    atomic_write_text(
        output_dir / "label_audit_v15.json",
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
    )

    write_text_audit(
        output_dir / "label_audit_v15.txt",
        audit,
    )

    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic SafeShell intent/risk labels."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input enriched JSONL. Defaults to enriched_commands_v7.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    if args.input is None:
        input_path = DEFAULT_INPUT
        if not input_path.exists() and V11_FALLBACK.exists():
            input_path = V11_FALLBACK
    else:
        input_path = args.input

    input_path = input_path.resolve()
    output_dir = args.output_dir.resolve()

    if not input_path.exists():
        raise SystemExit(
            f"Input file not found: {input_path}\n"
            "Pass the correct file with --input."
        )

    print("=" * 72)
    print("SafeShell Labels V15")
    print("=" * 72)
    print(f"Input       : {input_path}")
    print(f"Output      : {output_dir}")
    print(f"Label       : {LABEL_VERSION}")

    audit = run(input_path, output_dir)

    print(f"Total       : {audit['records_total']}")
    print(f"Training    : {audit['training_records']}")
    print(f"Review      : {audit['review_records']}")
    print(f"Malformed   : {audit['malformed_records']}")
    print(f"Duplicates  : {len(audit['duplicate_ids'])}")
    print(
        f"Conflicts   : {len(audit['duplicate_command_conflicts'])}"
    )
    print(
        f"Semantic flags: {sum(audit['semantic_flags'].values())}"
    )
    print(f"Status      : {audit['status']}")
    print("=" * 72)

    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()