"""
SafeShell Baseline V1
---------------------

Fast CPU baseline:
  command -> character TF-IDF -> LinearSVC

Two independent targets:
  1. intent: multi-label
  2. risk: 5-class

Evaluates:
  - IID test
  - grouped test

Outputs:
  models/baseline/
    intent_tfidf.joblib
    risk_tfidf.joblib
    metadata.json

  reports/baseline/
    baseline_report.json
    intent_report_iid.txt
    intent_report_grouped.txt
    risk_report_iid.txt
    risk_report_grouped.txt
    risk_confusion_iid.json
    risk_confusion_grouped.json

This intentionally uses the raw command only. It does NOT use:
  operation, operation_detail, domain, risk_flags, intent, risk,
  label_metadata, or other ontology-derived target information.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import joblib
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.svm import LinearSVC
except ImportError as exc:
    print(f"Missing dependency: {exc}")
    print("Install with:")
    print("  pip install scikit-learn joblib numpy")
    raise SystemExit(1)


DEFAULT_DATASET = Path("/home/paras/Desktop/SafeShell/data/dataset")
DEFAULT_MODEL_DIR = Path("/home/paras/Desktop/SafeShell/models/baseline")
DEFAULT_REPORT_DIR = Path("/home/paras/Desktop/SafeShell/reports/baseline")

RISK_LABELS = ["safe", "low", "medium", "high", "critical"]

FORBIDDEN_FEATURE_NAMES = {
    "intent",
    "risk",
    "operation",
    "operation_detail",
    "domain",
    "domain_action",
    "risk_features",
    "risk_flags",
    "label_metadata",
    "intent_resolution",
    "risk_reasons",
}


def get_nested(obj: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def extract_command(record: dict[str, Any]) -> str:
    paths = (
        ("command",),
        ("raw_command",),
        ("model_input", "command"),
        ("model_input", "raw_command"),
        ("input", "command"),
        ("parsed", "command"),
    )

    candidates = []
    for path in paths:
        value = get_nested(record, path)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    if not candidates:
        raise ValueError(f"Record {record.get('id')} has no command")

    normalized = {re.sub(r"\s+", " ", x) for x in candidates}
    if len(normalized) != 1:
        raise ValueError(
            f"Record {record.get('id')} has conflicting command fields"
        )

    return candidates[0]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no} is not an object")
            records.append(obj)
    return records


def validate_model_input(record: dict[str, Any]) -> list[str]:
    hits = []

    def walk(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                key = str(key)
                if key in FORBIDDEN_FEATURE_NAMES:
                    hits.append(f"{path}.{key}")
                walk(value, f"{path}.{key}")
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                walk(value, f"{path}[{i}]")

    walk(record.get("model_input", {}), "model_input")
    return hits


def labels(record: dict[str, Any]) -> tuple[list[str], str]:
    target = record.get("labels")
    if not isinstance(target, dict):
        raise ValueError(f"Record {record.get('id')} has no labels")

    intents = target.get("intent")
    risk = str(target.get("risk"))

    if isinstance(intents, str):
        intents = [intents]

    if not isinstance(intents, list) or not intents:
        raise ValueError(f"Invalid intent for {record.get('id')}")

    if risk not in RISK_LABELS:
        raise ValueError(f"Invalid risk {risk!r} for {record.get('id')}")

    return sorted(set(str(x) for x in intents)), risk


def load_split(path: Path):
    records = load_jsonl(path)
    commands = []
    intents = []
    risks = []

    for record in records:
        # Hard leakage check before feature extraction.
        leakage = validate_model_input(record)
        if leakage:
            raise RuntimeError(
                f"Target leakage found in {path}: {leakage[:10]}"
            )

        command = extract_command(record)
        intent, risk = labels(record)

        commands.append(command)
        intents.append(intent)
        risks.append(risk)

    return records, commands, intents, risks


def make_vectorizer() -> TfidfVectorizer:
    # Character n-grams are intentionally chosen for shell syntax:
    # flags, paths, punctuation, subcommands, operators, extensions, etc.
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        max_features=120_000,
        sublinear_tf=True,
        lowercase=False,
        dtype=np.float32,
    )


def train_intent(X, y):
    classifier = OneVsRestClassifier(
        LinearSVC(
            C=2.0,
            class_weight="balanced",
            dual="auto",
            max_iter=5000,
        ),
        n_jobs=-1,
    )
    classifier.fit(X, y)
    return classifier


def train_risk(X, y):
    classifier = LinearSVC(
        C=1.5,
        class_weight="balanced",
        dual="auto",
        max_iter=5000,
    )
    classifier.fit(X, y)
    return classifier


def evaluate_intent(model, X, y_true, mlb):
    y_pred = model.predict(X)

    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    exact = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=mlb.classes_,
        zero_division=0,
    )

    # Human-readable prediction-level error counts.
    exact_misses = 0
    false_positive = 0
    false_negative = 0

    for truth, pred in zip(y_true, y_pred):
        if not np.array_equal(truth, pred):
            exact_misses += 1
        false_positive += int(np.sum((pred == 1) & (truth == 0)))
        false_negative += int(np.sum((pred == 0) & (truth == 1)))

    return {
        "micro_f1": float(micro),
        "macro_f1": float(macro),
        "weighted_f1": float(weighted),
        "exact_match_accuracy": float(exact),
        "exact_match_errors": exact_misses,
        "false_positive_labels": false_positive,
        "false_negative_labels": false_negative,
        "classification_report": report,
    }, y_pred


def evaluate_risk(model, X, y_true):
    y_pred = model.predict(X)

    macro = f1_score(
        y_true, y_pred, labels=RISK_LABELS, average="macro", zero_division=0
    )
    weighted = f1_score(
        y_true, y_pred, labels=RISK_LABELS, average="weighted", zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, labels=RISK_LABELS, average="macro", zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, labels=RISK_LABELS, average="macro", zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=RISK_LABELS,
        target_names=RISK_LABELS,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=RISK_LABELS)

    # Safety-critical metric:
    # actual medium/high/critical predicted as safe.
    dangerous = {"medium", "high", "critical"}
    dangerous_count = 0
    dangerous_to_safe = 0
    high_critical_count = 0
    high_critical_to_safe = 0

    for truth, pred in zip(y_true, y_pred):
        if truth in dangerous:
            dangerous_count += 1
            if pred == "safe":
                dangerous_to_safe += 1

        if truth in {"high", "critical"}:
            high_critical_count += 1
            if pred == "safe":
                high_critical_to_safe += 1

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro),
        "weighted_f1": float(weighted),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "dangerous_to_safe_rate": (
            dangerous_to_safe / dangerous_count
            if dangerous_count else 0.0
        ),
        "dangerous_to_safe_count": dangerous_to_safe,
        "dangerous_count": dangerous_count,
        "high_critical_to_safe_rate": (
            high_critical_to_safe / high_critical_count
            if high_critical_count else 0.0
        ),
        "high_critical_to_safe_count": high_critical_to_safe,
        "high_critical_count": high_critical_count,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_labels": RISK_LABELS,
    }, y_pred


def save_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()

    print("=" * 72)
    print("SafeShell Baseline V1")
    print("=" * 72)
    print(f"Dataset : {args.dataset_dir.resolve()}")
    print(f"Models  : {args.model_dir.resolve()}")
    print(f"Reports : {args.report_dir.resolve()}")

    required = [
        "train.jsonl",
        "val.jsonl",
        "test_iid.jsonl",
        "test_grouped.jsonl",
    ]
    missing = [
        name for name in required
        if not (args.dataset_dir / name).exists()
    ]
    if missing:
        raise RuntimeError(
            "Missing dataset files: " + ", ".join(missing)
        )

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Load
    # ------------------------------------------------------------
    t0 = time.perf_counter()

    train_records, train_commands, train_intents, train_risks = load_split(
        args.dataset_dir / "train.jsonl"
    )
    val_records, val_commands, val_intents, val_risks = load_split(
        args.dataset_dir / "val.jsonl"
    )
    iid_records, iid_commands, iid_intents, iid_risks = load_split(
        args.dataset_dir / "test_iid.jsonl"
    )
    grouped_records, grouped_commands, grouped_intents, grouped_risks = load_split(
        args.dataset_dir / "test_grouped.jsonl"
    )

    print(
        f"Loaded: train={len(train_records)}, "
        f"val={len(val_records)}, "
        f"iid_test={len(iid_records)}, "
        f"grouped_test={len(grouped_records)}"
    )

    if not train_records or not val_records or not iid_records or not grouped_records:
        raise RuntimeError("One or more required splits are empty.")

    # ------------------------------------------------------------
    # Vectorizer
    # ------------------------------------------------------------
    vectorizer = make_vectorizer()
    X_train = vectorizer.fit_transform(train_commands)
    X_val = vectorizer.transform(val_commands)
    X_iid = vectorizer.transform(iid_commands)
    X_grouped = vectorizer.transform(grouped_commands)

    print(f"Features: {X_train.shape[1]:,}")
    print(f"Train matrix: {X_train.shape[0]:,} x {X_train.shape[1]:,}")

    # ------------------------------------------------------------
    # Intent model
    # ------------------------------------------------------------
    mlb = MultiLabelBinarizer()
    y_train_intent = mlb.fit_transform(train_intents)
    y_val_intent = mlb.transform(val_intents)
    y_iid_intent = mlb.transform(iid_intents)
    y_grouped_intent = mlb.transform(grouped_intents)

    print(f"Intent labels: {len(mlb.classes_)}")

    intent_model = train_intent(X_train, y_train_intent)

    val_intent_metrics, _ = evaluate_intent(
        intent_model, X_val, y_val_intent, mlb
    )
    iid_intent_metrics, iid_intent_pred = evaluate_intent(
        intent_model, X_iid, y_iid_intent, mlb
    )
    grouped_intent_metrics, grouped_intent_pred = evaluate_intent(
        intent_model, X_grouped, y_grouped_intent, mlb
    )

    # ------------------------------------------------------------
    # Risk model
    # ------------------------------------------------------------
    risk_model = train_risk(X_train, train_risks)

    val_risk_metrics, _ = evaluate_risk(
        risk_model, X_val, val_risks
    )
    iid_risk_metrics, iid_risk_pred = evaluate_risk(
        risk_model, X_iid, iid_risks
    )
    grouped_risk_metrics, grouped_risk_pred = evaluate_risk(
        risk_model, X_grouped, grouped_risks
    )

    # ------------------------------------------------------------
    # Save models
    # ------------------------------------------------------------
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "model": intent_model,
            "label_binarizer": mlb,
            "feature_type": "character_tfidf",
            "ngram_range": (2, 5),
        },
        args.model_dir / "intent_tfidf.joblib",
    )

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "model": risk_model,
            "risk_labels": RISK_LABELS,
            "feature_type": "character_tfidf",
            "ngram_range": (2, 5),
        },
        args.model_dir / "risk_tfidf.joblib",
    )

    metadata = {
        "model": "LinearSVC",
        "feature_extractor": "TfidfVectorizer",
        "analyzer": "char",
        "ngram_range": [2, 5],
        "max_features": 120000,
        "lowercase": False,
        "intent": {
            "type": "multi_label",
            "labels": mlb.classes_.tolist(),
            "classifier": "OneVsRest(LinearSVC)",
        },
        "risk": {
            "type": "multiclass",
            "labels": RISK_LABELS,
            "classifier": "LinearSVC",
        },
        "training_records": len(train_records),
        "validation_records": len(val_records),
        "iid_test_records": len(iid_records),
        "grouped_test_records": len(grouped_records),
        "feature_count": int(X_train.shape[1]),
    }
    save_json(args.model_dir / "metadata.json", metadata)

    # ------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------
    report = {
        "status": "PASS",
        "model": metadata,
        "validation": {
            "intent": val_intent_metrics,
            "risk": val_risk_metrics,
        },
        "iid_test": {
            "intent": iid_intent_metrics,
            "risk": iid_risk_metrics,
        },
        "grouped_test": {
            "intent": grouped_intent_metrics,
            "risk": grouped_risk_metrics,
        },
        "runtime_seconds": round(time.perf_counter() - t0, 4),
    }

    save_json(args.report_dir / "baseline_report.json", report)

    for name, metrics in (
        ("intent_report_iid.txt", iid_intent_metrics["classification_report"]),
        ("intent_report_grouped.txt", grouped_intent_metrics["classification_report"]),
        ("risk_report_iid.txt", iid_risk_metrics["classification_report"]),
        ("risk_report_grouped.txt", grouped_risk_metrics["classification_report"]),
    ):
        (args.report_dir / name).write_text(metrics, encoding="utf-8")

    save_json(
        args.report_dir / "risk_confusion_iid.json",
        {
            "labels": RISK_LABELS,
            "matrix": iid_risk_metrics["confusion_matrix"],
        },
    )
    save_json(
        args.report_dir / "risk_confusion_grouped.json",
        {
            "labels": RISK_LABELS,
            "matrix": grouped_risk_metrics["confusion_matrix"],
        },
    )

    # ------------------------------------------------------------
    # Terminal summary
    # ------------------------------------------------------------
    print("\n" + "=" * 72)
    print("BASELINE RESULTS")
    print("=" * 72)

    print("\nValidation:")
    print(
        f"  Intent micro-F1 : {val_intent_metrics['micro_f1']:.4f}"
    )
    print(
        f"  Intent macro-F1 : {val_intent_metrics['macro_f1']:.4f}"
    )
    print(
        f"  Risk macro-F1   : {val_risk_metrics['macro_f1']:.4f}"
    )
    print(
        f"  Risk accuracy   : {val_risk_metrics['accuracy']:.4f}"
    )

    print("\nIID test:")
    print(
        f"  Intent micro-F1 : {iid_intent_metrics['micro_f1']:.4f}"
    )
    print(
        f"  Intent macro-F1 : {iid_intent_metrics['macro_f1']:.4f}"
    )
    print(
        f"  Intent exact    : {iid_intent_metrics['exact_match_accuracy']:.4f}"
    )
    print(
        f"  Risk macro-F1   : {iid_risk_metrics['macro_f1']:.4f}"
    )
    print(
        f"  Risk accuracy   : {iid_risk_metrics['accuracy']:.4f}"
    )
    print(
        f"  Dangerous→safe  : "
        f"{iid_risk_metrics['dangerous_to_safe_rate']:.4f}"
    )

    print("\nGrouped test:")
    print(
        f"  Intent micro-F1 : {grouped_intent_metrics['micro_f1']:.4f}"
    )
    print(
        f"  Intent macro-F1 : {grouped_intent_metrics['macro_f1']:.4f}"
    )
    print(
        f"  Intent exact    : {grouped_intent_metrics['exact_match_accuracy']:.4f}"
    )
    print(
        f"  Risk macro-F1   : {grouped_risk_metrics['macro_f1']:.4f}"
    )
    print(
        f"  Risk accuracy   : {grouped_risk_metrics['accuracy']:.4f}"
    )
    print(
        f"  Dangerous→safe  : "
        f"{grouped_risk_metrics['dangerous_to_safe_rate']:.4f}"
    )

    print(f"\nRuntime: {report['runtime_seconds']:.2f}s")
    print(f"Report : {args.report_dir / 'baseline_report.json'}")
    print("STATUS: PASS")


if __name__ == "__main__":
    raise SystemExit(main())