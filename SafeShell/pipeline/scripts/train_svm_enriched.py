#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC


RISK_LABELS = ["safe", "low", "medium", "high", "critical"]
DANGEROUS_LABELS = {"medium", "high", "critical"}


# ============================================================
# LOAD
# ============================================================

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
                    f"Invalid JSON in {path}, line {line_no}: {exc}"
                ) from exc

    if not records:
        raise ValueError(f"No records found: {path}")

    return records


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def get_model_input(record):
    model_input = record.get("model_input")

    if not isinstance(model_input, dict):
        raise ValueError(
            f"Missing model_input in record {record.get('id')}"
        )

    return model_input


def get_command(record):
    model_input = get_model_input(record)

    command = model_input.get("command")

    if command is None:
        raise ValueError(
            f"Missing command in record {record.get('id')}"
        )

    return str(command)


def get_intents(record):
    labels = record.get("labels", {})
    intents = labels.get("intent", [])

    if isinstance(intents, str):
        intents = [intents]

    if not isinstance(intents, list):
        raise ValueError(
            f"Invalid intent labels in record {record.get('id')}"
        )

    return sorted(set(str(x) for x in intents))


def get_risk(record):
    risk = record.get("labels", {}).get("risk")

    if risk not in RISK_LABELS:
        raise ValueError(
            f"Invalid risk '{risk}' in record {record.get('id')}"
        )

    return risk


def list_to_text(value):
    if not isinstance(value, list):
        return ""

    return " ".join(str(x) for x in value)


def recursive_flatten(value, prefix="", output=None):
    """
    Convert nested dictionaries/lists into flat categorical/numeric
    features suitable for DictVectorizer.
    """

    if output is None:
        output = {}

    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = (
                f"{prefix}.{key}" if prefix else str(key)
            )

            recursive_flatten(
                child,
                child_prefix,
                output,
            )

    elif isinstance(value, list):

        if not value:
            output[f"{prefix}.__empty__"] = 1.0
            return output

        # Preserve individual list elements as categorical features.
        for item in value:

            if isinstance(item, (dict, list)):
                # Nested structures are recursively flattened.
                recursive_flatten(
                    item,
                    prefix,
                    output,
                )
            else:
                feature_name = (
                    f"{prefix}={str(item)}"
                )
                output[feature_name] = (
                    output.get(feature_name, 0.0) + 1.0
                )

        # Also preserve list length.
        output[f"{prefix}.__length__"] = float(len(value))

    elif isinstance(value, bool):
        output[prefix] = 1.0 if value else 0.0

    elif isinstance(value, (int, float)):
        output[prefix] = float(value)

    elif value is None:
        output[f"{prefix}=NULL"] = 1.0

    else:
        output[f"{prefix}={str(value)}"] = 1.0

    return output


def build_structured_features(record):
    """
    Extract only model_input.

    labels and label_metadata are deliberately excluded.
    """

    model_input = get_model_input(record)

    features = {}

    # Explicitly flatten the actual SafeShell model_input schema.
    for key, value in model_input.items():

        # Raw command is represented separately through TF-IDF.
        if key == "command":
            continue

        recursive_flatten(
            value,
            prefix=key,
            output=features,
        )

    return features


def prepare(records):
    commands = []
    intents = []
    risks = []
    structured = []

    for record in records:
        commands.append(get_command(record))
        intents.append(get_intents(record))
        risks.append(get_risk(record))
        structured.append(
            build_structured_features(record)
        )

    return (
        commands,
        intents,
        risks,
        structured,
    )


# ============================================================
# MODELS
# ============================================================

def build_text_vectorizer():
    return TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
    )


def train_intent_model(X, y):
    return OneVsRestClassifier(
        LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        )
    ).fit(X, y)


def train_risk_model(X, y):
    return LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    ).fit(X, y)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_intent(model, X, y_true, mlb):
    y_pred = model.predict(X)

    return {
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
        "exact_match": float(
            np.mean(
                np.all(
                    y_true == y_pred,
                    axis=1,
                )
            )
        ),
        "hamming_loss": float(
            hamming_loss(y_true, y_pred)
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=mlb.classes_,
            output_dict=True,
            zero_division=0,
        ),
    }


def evaluate_risk(model, X, y_true):
    y_pred = model.predict(X)

    dangerous = np.isin(
        y_true,
        list(DANGEROUS_LABELS),
    )

    high_critical = np.isin(
        y_true,
        ["high", "critical"],
    )

    critical = y_true == "critical"

    dangerous_count = int(dangerous.sum())
    high_critical_count = int(high_critical.sum())
    critical_count = int(critical.sum())

    dangerous_to_safe = int(
        np.sum(
            dangerous & (y_pred == "safe")
        )
    )

    high_critical_to_safe = int(
        np.sum(
            high_critical & (y_pred == "safe")
        )
    )

    critical_to_safe = int(
        np.sum(
            critical & (y_pred == "safe")
        )
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=RISK_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=RISK_LABELS,
                average="weighted",
                zero_division=0,
            )
        ),
        "dangerous_to_safe_rate": (
            dangerous_to_safe / dangerous_count
            if dangerous_count
            else 0.0
        ),
        "dangerous_to_safe_count": dangerous_to_safe,
        "dangerous_count": dangerous_count,
        "high_or_critical_to_safe_rate": (
            high_critical_to_safe /
            high_critical_count
            if high_critical_count
            else 0.0
        ),
        "high_or_critical_to_safe_count":
            high_critical_to_safe,
        "high_or_critical_count":
            high_critical_count,
        "critical_to_safe_rate": (
            critical_to_safe / critical_count
            if critical_count
            else 0.0
        ),
        "critical_to_safe_count":
            critical_to_safe,
        "critical_count":
            critical_count,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=RISK_LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }


# ============================================================
# SPLIT EVALUATION
# ============================================================

def evaluate_split(
    name,
    records,
    text_vectorizer,
    structured_vectorizer,
    intent_model,
    risk_model,
    mlb,
):
    (
        commands,
        intents,
        risks,
        structured,
    ) = prepare(records)

    X_text = text_vectorizer.transform(
        commands
    )

    X_structured = structured_vectorizer.transform(
        structured
    )

    X = hstack(
        [
            X_text,
            X_structured,
        ],
        format="csr",
    )

    y_intent = mlb.transform(intents)
    y_risk = np.array(risks)

    return {
        "split": name,
        "records": len(records),
        "features": X.shape[1],
        "intent": evaluate_intent(
            intent_model,
            X,
            y_intent,
            mlb,
        ),
        "risk": evaluate_risk(
            risk_model,
            X,
            y_risk,
        ),
    }


# ============================================================
# PRINT
# ============================================================

def print_summary(result):
    print()
    print("=" * 72)
    print(result["split"].upper())
    print("=" * 72)

    print(
        f"Records             : "
        f"{result['records']}"
    )

    print(
        f"Features            : "
        f"{result['features']}"
    )

    intent = result["intent"]

    print("\nIntent:")
    print(
        f"  Micro-F1          : "
        f"{intent['micro_f1']:.4f}"
    )
    print(
        f"  Macro-F1          : "
        f"{intent['macro_f1']:.4f}"
    )
    print(
        f"  Weighted-F1       : "
        f"{intent['weighted_f1']:.4f}"
    )
    print(
        f"  Exact match       : "
        f"{intent['exact_match']:.4f}"
    )
    print(
        f"  Hamming loss      : "
        f"{intent['hamming_loss']:.4f}"
    )

    risk = result["risk"]

    print("\nRisk:")
    print(
        f"  Accuracy          : "
        f"{risk['accuracy']:.4f}"
    )
    print(
        f"  Macro-F1          : "
        f"{risk['macro_f1']:.4f}"
    )
    print(
        f"  Weighted-F1       : "
        f"{risk['weighted_f1']:.4f}"
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


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "SafeShell enriched "
            "TF-IDF + Linear SVM"
        )
    )

    parser.add_argument(
        "--dataset-dir",
        default="data/dataset",
    )

    parser.add_argument(
        "--model-dir",
        default="models/svm_enriched",
    )

    parser.add_argument(
        "--report-dir",
        default="reports/svm_enriched",
    )

    args = parser.parse_args()

    dataset_dir = Path(
        args.dataset_dir
    )

    model_dir = Path(
        args.model_dir
    )

    report_dir = Path(
        args.report_dir
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "train":
            dataset_dir / "train.jsonl",

        "validation":
            dataset_dir / "val.jsonl",

        "iid_test":
            dataset_dir / "test_iid.jsonl",

        "grouped_test":
            dataset_dir / "test_grouped.jsonl",
    }

    print("=" * 72)
    print("SafeShell Enriched Linear SVM V1")
    print("=" * 72)

    for name, path in paths.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {name} split: {path}"
            )

    train_records = load_jsonl(
        paths["train"]
    )

    validation_records = load_jsonl(
        paths["validation"]
    )

    iid_records = load_jsonl(
        paths["iid_test"]
    )

    grouped_records = load_jsonl(
        paths["grouped_test"]
    )

    print(
        f"Train records   : "
        f"{len(train_records)}"
    )

    print(
        f"Validation      : "
        f"{len(validation_records)}"
    )

    print(
        f"IID test        : "
        f"{len(iid_records)}"
    )

    print(
        f"Grouped test    : "
        f"{len(grouped_records)}"
    )

    (
        train_commands,
        train_intents,
        train_risks,
        train_structured,
    ) = prepare(train_records)

    # --------------------------------------------------------
    # Intent encoding
    # --------------------------------------------------------

    mlb = MultiLabelBinarizer()

    y_intent_train = mlb.fit_transform(
        train_intents
    )

    print(
        f"Intent labels   : "
        f"{len(mlb.classes_)}"
    )

    print(
        f"Labels          : "
        f"{', '.join(mlb.classes_)}"
    )

    # --------------------------------------------------------
    # TEXT FEATURES
    # --------------------------------------------------------

    print(
        "\nFitting WORD TF-IDF "
        "on TRAIN ONLY..."
    )

    text_vectorizer = (
        build_text_vectorizer()
    )

    X_text_train = (
        text_vectorizer.fit_transform(
            train_commands
        )
    )

    print(
        f"Text features   : "
        f"{X_text_train.shape[1]}"
    )

    # --------------------------------------------------------
    # STRUCTURED FEATURES
    # --------------------------------------------------------

    print(
        "Fitting STRUCTURED "
        "features on TRAIN ONLY..."
    )

    structured_vectorizer = (
        DictVectorizer(
            sparse=True,
            dtype=np.float64,
        )
    )

    X_structured_train = (
        structured_vectorizer.fit_transform(
            train_structured
        )
    )

    print(
        f"Structured features : "
        f"{X_structured_train.shape[1]}"
    )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    X_train = hstack(
        [
            X_text_train,
            X_structured_train,
        ],
        format="csr",
    )

    print(
        f"Combined features : "
        f"{X_train.shape[1]}"
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print(
        "\nTraining intent "
        "Linear SVM..."
    )

    intent_model = train_intent_model(
        X_train,
        y_intent_train,
    )

    print(
        "Training risk "
        "Linear SVM..."
    )

    risk_model = train_risk_model(
        X_train,
        np.array(train_risks),
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    joblib.dump(
        text_vectorizer,
        model_dir / "text_tfidf.joblib",
    )

    joblib.dump(
        structured_vectorizer,
        model_dir / "structured_vectorizer.joblib",
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

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    results = []

    for name, records in [
        (
            "validation",
            validation_records,
        ),
        (
            "iid_test",
            iid_records,
        ),
        (
            "grouped_test",
            grouped_records,
        ),
    ]:

        result = evaluate_split(
            name,
            records,
            text_vectorizer,
            structured_vectorizer,
            intent_model,
            risk_model,
            mlb,
        )

        results.append(result)

        print_summary(result)

        with open(
            report_dir /
            f"{name}_report.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                result,
                f,
                indent=2,
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = {
        "model":
            "linear_svm_enriched",

        "feature_type":
            "word_tfidf_plus_structured",

        "text_features": {
            "analyzer": "word",
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.98,
            "sublinear_tf": True,
        },

        "structured_features":
            "model_input_only",

        "target_leakage_protection": {
            "labels_excluded": True,
            "label_metadata_excluded": True,
        },

        "intent": {
            "type": "multi_label",
            "classifier":
                "OneVsRest LinearSVC",
            "C": 1.0,
            "class_weight": "balanced",
            "labels":
                mlb.classes_.tolist(),
        },

        "risk": {
            "type": "multiclass",
            "classifier": "LinearSVC",
            "C": 1.0,
            "class_weight": "balanced",
            "labels": RISK_LABELS,
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

    print()
    print("=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)

    print(
        f"Models  : {model_dir}"
    )

    print(
        f"Reports : {report_dir}"
    )


if __name__ == "__main__":
    main()