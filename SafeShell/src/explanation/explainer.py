#!/usr/bin/env python3

import argparse
import json
import sys

import ollama


MODEL = "gemma3:1b"


SYSTEM_PROMPT = """You are SafeShell's security explanation module.

Your ONLY job is to explain the supplied security analysis and suggest a
safer approach in natural language.

Rules:
1. Use ONLY the supplied facts.
2. Never change or reinterpret the supplied risk, intent, or decision.
3. Do not invent threats, capabilities, or reasons.
4. Do not provide commands to execute.
5. Do not provide alternative commands.
6. Do not provide links or external resources.
7. Do not provide recovery instructions.
8. Do not add disclaimers.
9. Keep the explanation concise.
10. Return exactly TWO sections.
11. The first section must be named exactly: Explanation:
12. The second section must be named exactly: Safer approach:
13. The Safer approach must be natural-language guidance only.
14. Never put executable commands, code, shell syntax, or command examples
    inside either section.
15. If the command is already safe and no safer approach is needed, say:
    "No alternative approach is necessary because the supplied command is
    already safe."
"""


def generate_explanation(
    command,
    intent,
    risk,
    decision,
    reasons,
):
    payload = {
        "command": command,
        "intent": intent,
        "risk": risk,
        "decision": decision,
        "reasons": reasons,
    }

    user_prompt = f"""Explain this SafeShell security result.

Security analysis:
{json.dumps(payload, indent=2)}

Return ONLY these two sections:

Explanation:
<brief explanation of why the supplied decision makes sense>

Safer approach:
<brief natural-language safer approach; do NOT provide a command>

Do not include any other sections.
Do not provide commands or code.
"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        options={
            "temperature": 0.1,
            "num_predict": 120,
        },
    )

    return response["message"]["content"].strip()


def main():
    parser = argparse.ArgumentParser(
        description="SafeShell Gemma explanation module"
    )

    parser.add_argument(
        "--command",
        required=True,
    )

    parser.add_argument(
        "--intent",
        required=True,
    )

    parser.add_argument(
        "--risk",
        required=True,
        choices=[
            "safe",
            "low",
            "medium",
            "high",
            "critical",
        ],
    )

    parser.add_argument(
        "--decision",
        required=True,
    )

    parser.add_argument(
        "--reasons",
        default="",
        help="Comma-separated security reasons",
    )

    args = parser.parse_args()

    reasons = [
        reason.strip()
        for reason in args.reasons.split(",")
        if reason.strip()
    ]

    try:
        explanation = generate_explanation(
            command=args.command,
            intent=args.intent,
            risk=args.risk,
            decision=args.decision,
            reasons=reasons,
        )

        print()
        print(explanation)

    except Exception as exc:
        print(
            f"Explanation model error: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()