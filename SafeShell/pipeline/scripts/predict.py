#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack

# Make scripts/ imports work
_SAFESHELL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SAFESHELL_ROOT / "pipeline" / "scripts"))
sys.path.insert(0, str(_SAFESHELL_ROOT))
from enrich import enrich_record
from src.explanation.explainer import generate_explanation

MODEL_DIR = _SAFESHELL_ROOT / "pipeline" / "models" / "svm_enriched"

RISK_ORDER = {
    "safe": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    text_vectorizer = joblib.load(
        MODEL_DIR / "text_tfidf.joblib"
    )

    structured_vectorizer = joblib.load(
        MODEL_DIR / "structured_vectorizer.joblib"
    )

    intent_model = joblib.load(
        MODEL_DIR / "intent_model.joblib"
    )

    risk_model = joblib.load(
        MODEL_DIR / "risk_model.joblib"
    )

    label_binarizer = joblib.load(
        MODEL_DIR / "intent_label_binarizer.joblib"
    )

    return (
        text_vectorizer,
        structured_vectorizer,
        intent_model,
        risk_model,
        label_binarizer,
    )


# ============================================================
# HELPERS
# ============================================================

def unique(items):
    return list(dict.fromkeys(
        x for x in items
        if x is not None and x != ""
    ))


def get_words(command):
    """
    Lightweight lexical representation.

    The actual semantic/shell analysis is still performed by
    enrich.py. This is only used to construct model_input fields.
    """
    try:
        import shlex
        return shlex.split(command, posix=True)
    except (ValueError, TypeError):
        return command.split()


def get_arguments(words, program, flags):
    arguments = []

    seen_program = False

    for word in words:

        if not seen_program:

            if word == program:
                seen_program = True

            continue

        if word in flags:
            continue

        if word.startswith("-"):
            continue

        arguments.append(word)

    return unique(arguments)


def get_wrappers(words):
    wrappers = {
        "sudo", "doas", "env", "nohup",
        "timeout", "nice", "ionice",
        "chrt", "taskset", "stdbuf",
        "flock", "xargs",
    }

    return unique(
        word for word in words
        if word in wrappers
    )


def get_subcommand_sequence(enrichment):
    stages = enrichment.get(
        "command_structure", {}
    ).get("stages", [])

    return [
        stage.get("subcommand")
        for stage in stages
        if stage.get("subcommand")
    ]


def get_program_sequence(enrichment):
    programs = enrichment.get(
        "command_structure", {}
    ).get("programs", [])

    return programs if isinstance(
        programs, list
    ) else []


def build_model_input(command):
    """
    Convert the live command into the same general model_input
    schema used during training.

    Labels are NEVER included.
    """

    raw_record = {
        "id": 0,
        "command": command,
        "commands": get_words(command),
    }

    enriched_record = enrich_record(raw_record)
    enrichment = enriched_record["enrichment"]

    words = get_words(command)

    program = enrichment.get("program")
    subcommand = enrichment.get("subcommand")

    flags = enrichment.get("flags", [])
    operators = enrichment.get(
        "command_structure", {}
    ).get("operators", [])

    redirections = enrichment.get(
        "command_structure", {}
    ).get("redirections", [])

    paths = enrichment.get("paths", [])

    environment_variables = enrichment.get(
        "environment_variables", []
    )

    shell = enrichment.get(
        "shell_features", {}
    )

    risk = enrichment.get(
        "risk_features", {}
    )

    programs = get_program_sequence(
        enrichment
    )

    subcommands = get_subcommand_sequence(
        enrichment
    )

    wrappers = get_wrappers(words)

    arguments = get_arguments(
        words,
        program,
        flags,
    )

    model_input = {
        "command": command,

        "program": program,

        "raw_program": program,

        "program_type": enrichment.get(
            "program_type"
        ),

        "subcommand": subcommand,

        "program_sequence": programs,

        "subcommand_sequence": subcommands,

        "stage_count": enrichment.get(
            "command_structure",
            {}
        ).get(
            "pipeline_length",
            len(programs) or 1,
        ),

        "wrappers": wrappers,

        "wrapper_arguments": [],

        "commands": words,

        "flags": flags,

        "arguments": arguments,

        "operators": operators,

        "redirections": redirections,

        "paths": paths,

        "environment_variables":
            environment_variables,

        "has_sudo": (
            "sudo" in wrappers
            or risk.get("privileged", False)
        ),

        "has_pipe": any(
            op in {"|", "|&"}
            for op in operators
        ),

        "has_redirection": bool(
            redirections
        ),

        "has_chaining": any(
            op in {
                "&&", "||",
                ";", ";;",
                ";&", ";;&"
            }
            for op in operators
        ),

        "has_command_substitution":
            shell.get(
                "has_command_substitution",
                False,
            ),

        "has_variable_assignment":
            bool(
                enrichment.get(
                    "assignments",
                    []
                )
            ),

        "has_glob":
            shell.get(
                "has_glob",
                False,
            ),

        "has_shell_expansion":
            (
                shell.get(
                    "has_parameter_expansion",
                    False,
                )
                or shell.get(
                    "has_arithmetic_expansion",
                    False,
                )
            ),

        "has_quotes":
            shell.get(
                "has_quotes",
                False,
            ),

        "has_subshell":
            shell.get(
                "has_subshell",
                False,
            ),

        "execution_mode":
            "background"
            if shell.get(
                "has_background",
                False,
            )
            else "foreground",

        "argument_roles":
            enrichment.get(
                "argument_roles",
                []
            ),

        "command_structure":
            enrichment.get(
                "command_structure",
                {}
            ),

        "shell_features":
            shell,
    }

    return model_input, enrichment


# ============================================================
# DECISION
# ============================================================

def make_decision(risk):
    """
    Temporary baseline decision policy.

    This is intentionally separate from the ML models so that
    the actual SafeShell trust engine can replace it later.
    """

    if risk == "critical":
        return "BLOCK"

    if risk == "high":
        return "BLOCK"

    if risk == "medium":
        return "CONFIRM"

    return "ALLOW"


# ============================================================
# PREDICTION
# ============================================================

def predict(command):

    (
        text_vectorizer,
        structured_vectorizer,
        intent_model,
        risk_model,
        label_binarizer,
    ) = load_models()

    model_input, enrichment = (
        build_model_input(command)
    )

    # --------------------------------------------------------
    # TEXT FEATURES
    # --------------------------------------------------------

    X_text = text_vectorizer.transform(
        [command]
    )

    # --------------------------------------------------------
    # STRUCTURED FEATURES
    # --------------------------------------------------------

    structured_features = {}

    # Reuse the SAME flattening implementation
    # from train_svm_enriched.py.
    from train_svm_enriched import (
        build_structured_features,
    )

    temporary_record = {
        "id": 0,
        "model_input": model_input,
    }

    structured_features = (
        build_structured_features(
            temporary_record
        )
    )

    X_structured = (
        structured_vectorizer.transform(
            [structured_features]
        )
    )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    X = hstack(
        [
            X_text,
            X_structured,
        ],
        format="csr",
    )

    # --------------------------------------------------------
    # INTENT
    # --------------------------------------------------------

    intent_prediction = (
        intent_model.predict(X)
    )

    intent_labels = label_binarizer.inverse_transform(
        intent_prediction
    )[0]

    intents = list(intent_labels)

    # Safety fallback
    if not intents:
        intents = ["unknown"]

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    risk = risk_model.predict(X)[0]

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    decision = make_decision(risk)

    # --------------------------------------------------------
    # REASONS
    # --------------------------------------------------------

    reasons = []

    risk_features = enrichment.get(
        "risk_features",
        {}
    )

    if risk_features.get("privileged"):
        reasons.append(
            "privilege_use"
        )

    if risk_features.get("destructive"):
        reasons.append(
            "destructive_action"
        )

    if risk_features.get("modifies_data"):
        reasons.append(
            "data_modification"
        )

    if risk_features.get(
        "modifies_permissions"
    ):
        reasons.append(
            "permission_or_identity_change"
        )

    if risk_features.get(
        "security_sensitive"
    ):
        reasons.append(
            "security_sensitive_operation"
        )

    if risk_features.get(
        "network_operation"
    ):
        reasons.append(
            "network_operation"
        )

    if risk_features.get(
        "execution_side_effect"
    ):
        reasons.append(
            "state_changing_execution"
        )

    reasons = unique(reasons)

    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    explanation = generate_explanation(
        command=command,
        intent=", ".join(intents),
        risk=risk,
        decision=decision,
        reasons=reasons,
    )

    return {
        "command": command,
        "intent": intents,
        "risk": risk,
        "decision": decision,
        "reasons": reasons,
        "explanation": explanation,
    }


# ============================================================
# PRINT
# ============================================================

def print_result(result):

    print()
    print("=" * 68)
    print("SafeShell")
    print("=" * 68)

    print(
        f"Command : {result['command']}"
    )

    print(
        f"Intent  : "
        f"{', '.join(result['intent'])}"
    )

    print(
        f"Risk    : {result['risk']}"
    )

    print(
        f"Decision: {result['decision']}"
    )

    if result["reasons"]:
        print(
            f"Reasons : "
            f"{', '.join(result['reasons'])}"
        )

    print()
    print("Explanation:")
    print(result["explanation"])

    print("=" * 68)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "SafeShell command inference"
        )
    )

    parser.add_argument(
        "command",
        nargs="+",
        help="Linux shell command to analyze",
    )

    args = parser.parse_args()

    command = " ".join(args.command)

    try:

        result = predict(command)

        print_result(result)

    except KeyboardInterrupt:

        print(
            "\nInterrupted.",
            file=sys.stderr,
        )

        sys.exit(130)

    except Exception as exc:

        print(
            f"SafeShell error: {exc}",
            file=sys.stderr,
        )

        sys.exit(1)


if __name__ == "__main__":
    main()