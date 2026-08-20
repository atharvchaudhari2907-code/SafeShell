#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    multilabel_confusion_matrix,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


RISK_ORDER = ["safe", "low", "medium", "high", "critical"]
DANGEROUS_RISKS = {"medium", "high", "critical"}


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_no}: {exc}"
                ) from exc

    if not records:
        raise ValueError(f"No records found in {path}")

    return records


def get_command(record):
    command = record.get("command")

    if command is None:
        command = record.get("model_input", {}).get("command")

    if command is None:
        raise ValueError(
            f"Record {record.get('id', '<unknown>')} has no command field"
        )

    return str(command)


def get_intents(record):
    labels = record.get("labels", {})
    intents = labels.get("intent", [])

    if isinstance(intents, str):
        intents = [intents]

    if not isinstance(intents, list):
        raise ValueError(
            f"Invalid intent labels in record {record.get('id', '<unknown>')}"
        )

    return sorted(set(str(x) for x in intents))


def get_risk(record):
    labels = record.get("labels", {})
    risk = labels.get("risk")

    if risk not in RISK_ORDER:
        raise ValueError(
            f"Invalid risk '{risk}' in record {record.get('id', '<unknown>')}"
        )

    return risk


def prepare(records):
    commands = [get_command(r) for r in records]
    intents = [get_intents(r) for r in records]
    risks = [get_risk(r) for r in records]

    return commands, intents, risks


def build_vectorizer():
    return TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
    )


def train_intent(X_train, y_train):
    model = OneVsRestClassifier(
        LogisticRegression(
            C=2.0,
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        )
    )

    model.fit(X_train, y_train)
    return model


def train_risk(X_train, y_train):
    model = LogisticRegression(
        C=2.0,
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="auto",
        random_state=42,
    )

    model.fit(X_train, y_train)
    return model


def evaluate_intent(model, X, y_true, mlb):
    y_pred = model.predict(X)

    result = {
        "micro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "exact_match": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=mlb.classes_,
            output_dict=True,
            zero_division=0,
        ),
    }

    return result


def evaluate_risk(model, X, y_true):
    y_pred = model.predict(X)

    dangerous_mask = np.isin(y_true, list(DANGEROUS_RISKS))

    dangerous_to_safe = float(
        np.mean((y_pred == "safe") & dangerous_mask)
    ) if np.any(dangerous_mask) else 0.0

    dangerous_count = int(np.sum(dangerous_mask))

    dangerous_to_safe_count = int(
        np.sum((y_pred == "safe") & dangerous_mask)
    )

    high_or_critical = np.isin(y_true, ["high", "critical"])
    high_or_critical_count = int(np.sum(high_or_critical))

    high_or_critical_to_safe = int(
        np.sum((y_pred == "safe") & high_or_critical)
    )

    critical = y_true == "critical"
    critical_count = int(np.sum(critical))

    critical_to_safe = int(
        np.sum((y_pred == "safe") & critical)
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=RISK_ORDER,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=RISK_ORDER,
                average="weighted",
                zero_division=0,
            )
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=RISK_ORDER,
            output_dict=True,
            zero_division=0,
        ),
        "dangerous_to_safe_rate": (
            dangerous_to_safe_count / dangerous_count
            if dangerous_count
            else 0.0
        ),
        "dangerous_to_safe_count": dangerous_to_safe_count,
        "dangerous_count": dangerous_count,
        "high_or_critical_to_safe_rate": (
            high_or_critical_to_safe / high_or_critical_count
            if high_or_critical_count
            else 0.0
        ),
        "high_or_critical_to_safe_count": high_or_critical_to_safe,
        "high_or_critical_count": high_or_critical_count,
        "critical_to_safe_rate": (
            critical_to_safe / critical_count
            if critical_count
            else 0.0
        ),
        "critical_to_safe_count": critical_to_safe,
        "critical_count": critical_count,
    }


def evaluate_split(
    name,
    records,
    vectorizer,
    intent_model,
    risk_model,
    mlb,
):
    commands, intents, risks = prepare(records)

    X = vectorizer.transform(commands)

    y_intent = mlb.transform(intents)
    y_risk = np.array(risks)

    intent_result = evaluate_intent(
        intent_model,
        X,
        y_intent,
        mlb,
    )

    risk_result = evaluate_risk(
        risk_model,
        X,
        y_risk,
    )

    return {
        "split": name,
        "records": len(records),
        "intent": intent_result,
        "risk": risk_result,
    }


def print_summary(result):
    print(f"\n{'=' * 72}")
    print(result["split"].upper())
    print(f"{'=' * 72}")

    print(
        f"Records             : {result['records']}"
    )

    intent = result["intent"]
    print("\nIntent:")
    print(
        f"  Micro-F1          : {intent['micro_f1']:.4f}"
    )
    print(
        f"  Macro-F1          : {intent['macro_f1']:.4f}"
    )
    print(
        f"  Weighted-F1       : {intent['weighted_f1']:.4f}"
    )
    print(
        f"  Exact match       : {intent['exact_match']:.4f}"
    )
    print(
        f"  Hamming loss      : {intent['hamming_loss']:.4f}"
    )

    risk = result["risk"]
    print("\nRisk:")
    print(
        f"  Accuracy          : {risk['accuracy']:.4f}"
    )
    print(
        f"  Macro-F1          : {risk['macro_f1']:.4f}"
    )
    print(
        f"  Weighted-F1       : {risk['weighted_f1']:.4f}"
    )
    print(
        f"  Dangerous -> safe : "
        f"{risk['dangerous_to_safe_rate']:.4f}"
    )
    print(
        f"  High/Critical -> safe : "
        f"{risk['high_or_critical_to_safe_rate']:.4f}"
    )
    print(
        f"  Critical -> safe  : "
        f"{risk['critical_to_safe_rate']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="SafeShell Logistic Regression baseline"
    )

    parser.add_argument(
        "--dataset-dir",
        default="data/dataset",
    )

    parser.add_argument(
        "--model-dir",
        default="models/logistic",
    )

    parser.add_argument(
        "--report-dir",
        default="reports/logistic_regression",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    model_dir = Path(args.model_dir)
    report_dir = Path(args.report_dir)

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": dataset_dir / "train.jsonl",
        "val": dataset_dir / "val.jsonl",
        "iid_test": dataset_dir / "test_iid.jsonl",
        "grouped_test": dataset_dir / "test_grouped.jsonl",
    }

    print("=" * 72)
    print("SafeShell Logistic Regression V1")
    print("=" * 72)

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {name} split: {path}"
            )

    train_records = load_jsonl(paths["train"])
    val_records = load_jsonl(paths["val"])
    iid_records = load_jsonl(paths["iid_test"])
    grouped_records = load_jsonl(paths["grouped_test"])

    print(f"Train records   : {len(train_records)}")
    print(f"Validation      : {len(val_records)}")
    print(f"IID test        : {len(iid_records)}")
    print(f"Grouped test    : {len(grouped_records)}")

    train_commands, train_intents, train_risks = prepare(
        train_records
    )

    mlb = MultiLabelBinarizer()
    y_intent_train = mlb.fit_transform(train_intents)

    print(f"Intent labels   : {len(mlb.classes_)}")
    print(
        f"Labels          : {', '.join(mlb.classes_)}"
    )

    vectorizer = build_vectorizer()

    print("\nFitting TF-IDF on TRAIN ONLY...")

    X_train = vectorizer.fit_transform(train_commands)

    print(
        f"TF-IDF matrix   : "
        f"{X_train.shape[0]} x {X_train.shape[1]}"
    )

    print("\nTraining intent Logistic Regression...")
    intent_model = train_intent(
        X_train,
        y_intent_train,
    )

    print("Training risk Logistic Regression...")
    risk_model = train_risk(
        X_train,
        np.array(train_risks),
    )

    # Save model components.
    joblib.dump(
        vectorizer,
        model_dir / "tfidf_vectorizer.joblib",
    )

    joblib.dump(
        intent_model,
        model_dir / "intent_model.joblib",
    )

    joblib.dump(
        risk_model,
        model_dir / "risk_model.joblib",
    )

    joblib.dump(
        mlb,
        model_dir / "intent_label_binarizer.joblib",
    )

    results = []

    for name, records in [
        ("validation", val_records),
        ("iid_test", iid_records),
        ("grouped_test", grouped_records),
    ]:
        result = evaluate_split(
            name,
            records,
            vectorizer,
            intent_model,
            risk_model,
            mlb,
        )

        results.append(result)
        print_summary(result)

        with open(
            report_dir / f"{name}_report.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                result,
                f,
                indent=2,
            )

    summary = {
        "model": "logistic_regression",
        "feature_type": "word_tfidf",
        "tfidf": {
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.98,
            "sublinear_tf": True,
        },
        "intent": {
            "type": "multi_label",
            "classifier": "OneVsRest LogisticRegression",
            "labels": mlb.classes_.tolist(),
        },
        "risk": {
            "type": "multiclass",
            "classifier": "LogisticRegression",
            "labels": RISK_ORDER,
        },
        "results": results,
    }

    with open(
        report_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print(f"\n{'=' * 72}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 72}")
    print(f"Models  : {model_dir}")
    print(f"Reports : {report_dir}")


if __name__ == "__main__":
    main()