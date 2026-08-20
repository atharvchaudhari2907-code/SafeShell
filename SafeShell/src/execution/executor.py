"""
executor.py

Command execution layer for SafeShell.

ONLY place that actually executes a command. Runs AFTER
CommandGateway + trust_engine. Does NOT re-implement safety itself.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

_SAFESHELL_ROOT = Path(__file__).resolve().parents[2]
if str(_SAFESHELL_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAFESHELL_ROOT))

from src.gateway.command_gateway import CommandGateway
from src.trust.trust_engine import evaluate, record_outcome
from src.trust.user_profile import get_user_profile
from src.audit.audit_logger import log_command


def execute_command(raw_command: str, role: str, user_id: str = "default", dry_run: bool = False) -> None:
    gateway = CommandGateway()
    profile = get_user_profile(role)
    print(f"User: {profile['name']}")

    result = gateway.process(raw_command)
    if result["status"] == "error":
        print(f"PARSE ERROR [{result['error_type']}]: {result['message']}")
        log_command(raw_command, role, profile["risk_level"], "ERROR", result["message"])
        return

    parsed_command = result["parsed_command"]
    analysis = result["analysis"]
    print(f"Risk: {analysis.get('final_risk','?').upper()} | Action: {analysis.get('action','?')}")
    print(f"Explanation: {analysis.get('explanation','-')}")

    decision = evaluate(raw_command, role, user_id, analysis, parsed_command)
    print(f"Decision: {decision['final_action']} -- {decision['reason']}")

    final_action = decision["final_action"]

    if final_action == "BLOCK":
        log_command(raw_command, role, profile["risk_level"], "BLOCKED", decision["reason"])
        record_outcome(user_id, parsed_command, "BLOCKED", analysis.get("final_risk","?"))
        return

    if final_action == "CONFIRM" and not decision["skip_confirmation"] and not dry_run:
        approval = input("Proceed anyway? (yes/no): ")
        if approval.strip().lower() != "yes":
            print("Rejected by user.")
            log_command(raw_command, role, profile["risk_level"], "REJECTED", "User declined")
            record_outcome(user_id, parsed_command, "REJECTED", analysis.get("final_risk","?"))
            return

    if dry_run:
        print(f"[DRY RUN] Would execute: {raw_command}")
        print(f"[DRY RUN] Decision: {final_action} -- command NOT actually run")
        log_command(raw_command, role, profile["risk_level"], "DRY_RUN", decision["reason"])
        record_outcome(user_id, parsed_command, "DRY_RUN", analysis.get("final_risk","?"))
        return

    _run(raw_command, role, profile["risk_level"], user_id, parsed_command, analysis)


def _run(raw_command, role, risk_level, user_id, parsed_command, analysis) -> None:
    try:
        result = subprocess.run(shlex.split(raw_command), capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        log_command(raw_command, role, risk_level, "ALLOWED", "Executed successfully")
        record_outcome(user_id, parsed_command, "ALLOWED", analysis.get("final_risk","?"))
    except Exception as exc:
        print("Execution error:", exc)
        log_command(raw_command, role, risk_level, "ERROR", str(exc))


if __name__ == "__main__":
    role_input = input("Enter user role (normal/developer/admin): ")
    user_id_input = input("Enter user id: ")
    command_input = input("Enter command: ")
    dry_run_input = input("Dry run? (yes/no): ").strip().lower() == "yes"
    execute_command(command_input, role_input, user_id_input, dry_run=dry_run_input)