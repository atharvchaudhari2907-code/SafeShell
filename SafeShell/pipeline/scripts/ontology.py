"""
SafeShell Ontology V4
---------------------

Purpose
-------
Audit the V11 enriched SafeShell dataset and produce a stable, compact
intent ontology for supervised training.

Important design decisions
--------------------------
1. Intent is MULTI-LABEL because shell commands can contain multiple
   semantic actions (for example: find + -exec).
2. operation_detail is preferred over the coarse operation whenever it
   resolves the meaning.
3. Risk is NOT derived from intent alone. V7 risk_features remain the
   source for the later risk-labeling stage.
4. Unknown/custom commands are retained in the audit but are NOT forced
   into a made-up intent.
5. This script is read-only with respect to the V7 dataset.

Input
-----
data/enriched/enriched_commands_v11.jsonl

Outputs
-------
data/ontology/ontology_audit_v4.json
data/ontology/ontology_audit_v4.txt
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = BASE_DIR / "data" / "enriched" / "enriched_commands_v11.jsonl"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "ontology"

ONTOLOGY_VERSION = "4.0"

# ---------------------------------------------------------------------------
# FINAL COMPACT INTENT ONTOLOGY
# ---------------------------------------------------------------------------

INTENTS = [
    "search",
    "inspect",
    "execute",
    "extract",
    "read",
    "manage_permissions",
    "manage_container",
    "list",
    "sort",
    "shell_control",
    "version_control",
    "copy",
    "connect",
    "manage_package",
    "transform",
    "filter",
    "move",
    "delete",
    "transfer",
    "manage_network",
    "manage_process",
    "install",
    "security_operation",
    "manage_identity",
    "uninstall",
    "manage_service",
    "manage_database",
    "manage_cloud",
    "modify",
    "configure",
    "create",
    "archive",
    "network_request",
    "compare",
    "output",
]

RISK_LEVELS = ["safe", "low", "medium", "high", "critical"]

# ---------------------------------------------------------------------------
# Stable operation -> intent mappings.
#
# These are used only where the operation itself has a stable semantic
# meaning. Mixed umbrella operations are handled separately below.
# ---------------------------------------------------------------------------

OPERATION_INTENTS: dict[str, list[str]] = {
    "search_text": ["search"],
    "search": ["search"],
    "inspect": ["inspect"],
    "inspect_usage": ["inspect"],
    "extract_fields": ["extract"],
    "extract": ["extract"],
    "read": ["read"],
    "list": ["list"],
    "sort": ["sort"],
    "change_permissions": ["manage_permissions"],
    "change_owner": ["manage_permissions"],
    "change_group": ["manage_permissions"],
    "copy": ["copy"],
    "move": ["move"],
    "delete": ["delete"],
    "output": ["output"],
    "split_output": ["output"],
    "formatted_output": ["output"],
    "transform_text": ["transform"],
    "transform_json": ["transform"],
    "transform": ["transform"],
    "filter_text": ["filter"],
    "modify_text": ["modify"],
    "filter": ["filter"],
    "container_operation": ["manage_container"],
    "kubernetes_operation": ["manage_container"],
    "version_control": ["version_control"],
    "network_configuration": ["manage_network"],
    "network_connection": ["connect"],
    "file_transfer": ["transfer"],
    "manage_service": ["manage_service"],
    "terminate_process": ["manage_process"],
    "replace_process": ["manage_process"],
    "process_control": ["manage_process"],
    "manage_process": ["manage_process"],
    "manage_group": ["manage_identity"],
    "manage_user": ["manage_identity"],
    "change_group": ["manage_identity"],
    "inspect_identity": ["inspect"],
    "identity_lookup": ["inspect"],
    "inspect_users": ["inspect"],
    "inspect_groups": ["inspect"],
    "manage_package_state": ["manage_package"],
    "install": ["install"],
    "uninstall": ["uninstall"],
    "inspect_package": ["inspect"],
    "cloud_operation": ["manage_cloud"],
    "database_operation": ["manage_database"],
    "database_backup": ["manage_database"],
    "database_restore": ["manage_database"],
    "cryptographic_operation": ["security_operation"],
    "key_management": ["security_operation"],
    "security_status": ["security_operation"],
    "security_configuration": ["security_operation"],
    "audit_management": ["security_operation"],
    "audit_search": ["security_operation"],
    "audit_report": ["security_operation"],
    "firewall_management": ["manage_network"],
    "packet_capture": ["manage_network"],
    "dns_configuration": ["manage_network"],
    "dns_lookup": ["inspect"],
    "route_diagnostic": ["inspect"],
    "connectivity_test": ["inspect"],
    "inspect_connections": ["inspect"],
    "inspect_interface": ["inspect"],
    "inspect_devices": ["inspect"],
    "inspect_capabilities": ["inspect"],
    "capability_inspection": ["inspect"],
    "inspect_location": ["inspect"],
    "inspect_logs": ["inspect"],
    "inspect_processes": ["inspect"],
    "search_processes": ["search"],
    "inspect_environment": ["inspect"],
    "inspect_resources": ["inspect"],
    "inspect_storage": ["inspect"],
    "inspect_mounts": ["inspect"],
    "inspect_sessions": ["inspect"],
    "inspect_login_history": ["inspect"],
    "inspect_scheduled_tasks": ["inspect"],
    "inspect_jobs": ["inspect"],
    "inspect_inventory": ["inspect"],
    "inspect_permissions": ["inspect"],
    "inspect_system": ["inspect"],
    "inspect_command": ["inspect"],
    "inspect_identity": ["inspect"],
    "checksum": ["inspect"],
    "partition": ["inspect"],
    "filesystem_check": ["inspect"],
    "filesystem_configuration": ["configure"],
    "mount": ["configure"],
    "unmount": ["configure"],
    "link": ["modify"],
    "archive": ["archive"],
    "compress": ["archive"],
    "decompress": ["archive"],
    "sync": ["transfer"],
    "create": ["create"],
    "generate_sequence": ["create"],
    "combine_columns": ["transform"],
    "join_records": ["transform"],
    "split_file": ["transform"],
    "deduplicate": ["transform"],
    "compare": ["compare"],
    "count": ["inspect"],
    "read_input": ["read"],
    "source_script": ["execute"],
    "execute_program": ["execute"],
    "execute_package_binary": ["execute"],
    "compile": ["execute"],
    "build": ["execute"],
    "format_code": ["execute"],
    "lint_code": ["execute"],
    "type_check": ["execute"],
    "evaluate_command": ["execute"],
    "configuration_automation": ["execute"],
    "infrastructure_operation": ["execute"],
    "configure": ["configure"],
    "system_configuration": ["configure"],
    "security_configuration": ["security_operation"],
    "shell_configuration": ["shell_control"],
    "control_structure": ["shell_control"],
    "conditional_test": ["shell_control"],
    "set_environment": ["shell_control"],
    "set_variable": ["shell_control"],
    "unset_environment": ["shell_control"],
    "shell_parameter": ["shell_control"],
    "parse_options": ["shell_control"],
    "function_control": ["shell_control"],
    "shell_control": ["shell_control"],
    "shell_history": ["shell_control"],
    "change_directory": ["shell_control"],
    "return_status": ["shell_control"],
    "resume_job": ["shell_control"],
    "wait": ["shell_control"],
    "detach_job": ["shell_control"],
    "repeat_command": ["shell_control"],
    "privilege_wrapper": ["security_operation"],
    "privileged_edit": ["security_operation"],
    "change_capabilities": ["security_operation"],
    "account_lockout": ["manage_identity"],
    "change_priority": ["manage_process"],
    "signal_handler": ["manage_process"],
    "schedule_task": ["shell_control"],
    "delete_scheduled_task": ["shell_control"],
    "swap_management": ["configure"],
    "device_mapper": ["configure"],
    "raw_io": ["execute"],
    "test": ["execute"],
    "trim": ["transform"],
    "format": ["transform"],
}

# ---------------------------------------------------------------------------
# Detail-level mappings for intentionally broad V7 operations.
# ---------------------------------------------------------------------------

DETAIL_INTENTS: dict[str, list[str]] = {
    # search + execution is inherently multi-intent
    "filesystem_search_exec": ["search", "execute"],

    # Git
    "git_push": ["version_control"],
    "git_restore": ["version_control"],
    "git_merge": ["version_control"],
    "git_pull": ["version_control"],
    "git_add": ["version_control"],
    "git_fetch": ["version_control"],
    "git_rebase": ["version_control"],
    "git_commit": ["version_control"],
    "git_clone": ["version_control"],
    "git_reset": ["version_control"],

    # Containers / Kubernetes
    "docker_run": ["manage_container"],
    "docker_exec": ["manage_container"],
    "docker_start": ["manage_container"],
    "docker_restart": ["manage_container"],
    "docker_pull": ["manage_container"],
    "docker_stop": ["manage_container"],
    "docker_rm": ["manage_container"],
    "docker_rmi": ["manage_container"],
    "kubectl_exec": ["manage_container"],
    "kubectl_edit": ["manage_container"],
    "kubectl_apply": ["manage_container"],
    "kubectl_delete": ["manage_container"],

    # NPM
    "npm_install": ["install"],
    "npm_run": ["execute"],
    "npm_update": ["manage_package"],
    "npm_exec": ["execute"],

    # HTTP/network
    "http_request": ["network_request"],
    "http_post": ["network_request"],
    "http_put": ["network_request"],
    "http_delete": ["network_request"],
    "http_patch": ["network_request"],
    "http_download": ["transfer"],
    "download": ["transfer"],

    # Package management
    "apt_install": ["install"],
    "apt-get_install": ["install"],
    "apt_remove": ["uninstall"],
    "apt-get_remove": ["uninstall"],
    "apt_purge": ["uninstall"],
    "apt_autoremove": ["uninstall"],
    "apt_update": ["manage_package"],
    "apt_upgrade": ["manage_package"],
    "apt_full-upgrade": ["manage_package"],
    "apt-get_download": ["transfer"],

    # System inspection
    "date": ["inspect"],
    "hostname": ["inspect"],
    "uname": ["inspect"],
    "hostnamectl": ["inspect"],
    "systemd_analysis": ["inspect"],
    "uptime": ["inspect"],

    # Services
    "service_manager": ["manage_service"],
    "systemctl_restart": ["manage_service"],
    "systemctl_enable": ["manage_service"],
    "systemctl_start": ["manage_service"],
    "systemctl_stop": ["manage_service"],
    "systemctl_reload": ["manage_service"],
    "systemctl_disable": ["manage_service"],
    "systemctl_reset-failed": ["manage_service"],
    "systemctl_daemon-reload": ["manage_service"],
    "systemctl_mask": ["manage_service"],
    "systemctl_unmask": ["manage_service"],
    "loginctl": ["manage_service"],

    # Access
    "secure_shell": ["connect"],
    "sftp": ["connect", "transfer"],

    # Terraform / infrastructure
    "terraform_state": ["inspect"],
    "terraform_workspace": ["configure"],
    "terraform_init": ["configure"],
    "terraform_validate": ["execute"],
    "terraform_fmt": ["execute"],
    "terraform_plan": ["inspect"],
    "terraform_apply": ["execute"],
    "terraform_destroy": ["delete"],
    "terraform_output": ["inspect"],
    "terraform_show": ["inspect"],

    # Build tool details
    "make": ["execute"],
    "cargo": ["execute"],
    "go": ["execute"],
    "maven": ["execute"],
    "cmake": ["execute"],
    "gradle": ["execute"],
    "dotnet": ["execute"],

    # Security/audit
    "auditctl": ["security_operation"],
    "auditd": ["security_operation"],
    "selinux_mode": ["security_operation"],
    "selinux_policy": ["security_operation"],
    "apparmor_parser": ["security_operation"],

    # Partition inspection in this corpus
    "fdisk": ["inspect"],
    "parted": ["inspect"],
    "partprobe": ["inspect"],

    # Jobs
    "foreground_job": ["shell_control"],
    "background_job": ["shell_control"],

    # Checksums
    "sha256": ["inspect"],
    "sha512": ["inspect"],
    "sha1": ["inspect"],
    "md5": ["inspect"],
    "cksum": ["inspect"],

    # Shell builtins / lookup
    "command_builtin": ["inspect"],
    "command_lookup": ["inspect"],
    "type_builtin": ["inspect"],
    "whereis": ["inspect"],

    # Database
    "database_backup": ["manage_database"],
    "database_restore": ["manage_database"],

    # Generic configuration
    "time_configuration": ["configure"],
}


# ---------------------------------------------------------------------------
# Special package fallback
# ---------------------------------------------------------------------------

def package_detail_intents(detail: str | None, command: str) -> tuple[list[str] | None, str]:
    """Resolve package-management details that were too coarse in V7."""
    if detail in DETAIL_INTENTS:
        return DETAIL_INTENTS[detail], "detail"

    text = command.lower()

    # Read-only package queries should not become manage_package.
    if (
        re.search(r"\b(dpkg|apt|apt-get|npm|snap|flatpak)\b", text)
        and re.search(r"(?:\s|^)(-s|-L|-l|--list|list|show|status|info|view)\b", text)
    ):
        return ["inspect"], "package_query"

    # npm generic rows are ambiguous without a subcommand.
    if re.search(r"\bnpm\b", text):
        return ["manage_package"], "package_fallback"

    # Generic dpkg rows are package-management observations in this corpus.
    if re.search(r"\bdpkg\b", text):
        return ["inspect"], "package_fallback"

    if re.search(r"\b(snap|flatpak)\b", text):
        return ["manage_package"], "package_fallback"

    if re.search(r"\b(apt|apt-get)\b", text):
        return ["manage_package"], "package_fallback"

    return None, "unresolved"


# ---------------------------------------------------------------------------
# Intent resolution
# ---------------------------------------------------------------------------

def command_semantic_override(record: dict[str, Any]) -> list[str] | None:
    command = str(record.get("command", "")).strip()
    e = record.get("enrichment") or {}
    program = str(e.get("program") or "")
    detail = str(e.get("operation_detail") or "")
    if program == "git" and re.search(r"\bgit\s+branch\s+(?:-D|--delete\s+--force)\b", command):
        return ["version_control", "delete"]
    if program == "git" and detail == "git_clean":
        if re.search(r"(?:^|\s)(?:--dry-run|-n|-[^\s-]*n[^\s-]*)(?:\s|$)", command):
            return ["version_control", "inspect"]
        return ["version_control", "delete"]
    if program == "git" and detail == "git_reset" and "--hard" in command.split():
        return ["version_control", "modify"]
    if program == "git" and detail == "git_rm":
        return ["version_control", "delete"]
    if program in {"userdel", "groupdel"} and detail in {"delete_user", "delete_group"}:
        return ["manage_identity", "delete"]
    if program == "chgrp":
        return ["manage_permissions"]
    if program == "git" and re.search(r"\bgit\s+branch\s+(?:-d|--delete)\b", command):
        return ["version_control", "delete"]
    if program == "git" and re.search(r"\bgit\s+remote\s+(?:add|remove|rename|set-url)\b", command):
        return ["version_control", "configure"]
    if program == "git" and re.search(r"\bgit\s+(?:merge-base|diff-tree)\b", command):
        return ["inspect"]
    if program == "git" and re.search(r"\bgit\s+config\s+(?:--get(?:-all|-regexp)?|--list|-l|--show-origin|--show-scope)\b", command):
        return ["inspect"]
    if program == "git" and re.search(r"\bgit\s+stash\s+(?:list|show)\b", command):
        return ["inspect"]
    if program == "git" and re.search(r"\bgit\s+worktree\s+list\b", command):
        return ["inspect"]
    if program == "git" and re.search(r"\bgit\s+push\s+(?:[^\n]*\s)?(?:--force|--force-with-lease)\b", command):
        return ["version_control"]
    if program == "git" and re.search(r"\bgit\s+push\s+[^\n]*--delete\b", command):
        return ["version_control", "delete"]
    if program == "git" and re.search(r"\bgit\s+(?:add|commit|checkout|switch|merge|rebase|stash|cherry-pick|restore|remote\s+(?:add|remove|rename|set-url))\b", command):
        return ["version_control"]
    if re.search(r"\b(?:python|python3)\s+-m\s+pip\s+install\b", command) and "--dry-run" not in command:
        return ["install"]
    if re.search(r"\b(?:python|python3)\s+-m\s+pip\s+uninstall\b", command):
        return ["uninstall"]
    if program == "docker" and re.search(r"\bdocker\s+(?:container\s+)?kill\b", command):
        return ["manage_container", "manage_process"]
    if program == "docker" and re.search(r"\bdocker\s+(?:container\s+)?cp\b", command):
        return ["manage_container", "transfer"]
    if program == "kubectl" and re.search(r"\bkubectl\s+cp\b", command):
        return ["manage_container", "transfer"]
    if re.search(r"\baws\s+s3\s+cp\b|\bgcloud\s+storage\s+cp\b", command):
        return ["manage_cloud", "transfer"]
    if program == "docker" and re.search(r"\bdocker\s+compose\s+run\b", command):
        return ["manage_container", "execute"]
    if re.search(r"\bdocker\s+(?:container\s+)?(?:rm|rmi|prune)\b", command):
        return ["manage_container", "delete"]
    if re.search(r"\bdocker\s+(?:container\s+)?kill\b", command):
        return ["manage_container", "manage_process"]
    if re.search(r"\bdocker\s+network\s+(?:create|connect|disconnect)\b", command):
        return ["manage_container", "manage_network"]
    if re.search(r"\bdocker\s+volume\s+(?:create|rm|prune)\b", command):
        return ["manage_container", "create" if re.search(r"\bcreate\b", command) else "delete"]
    if re.search(r"\bkubectl\s+delete\b", command):
        return ["manage_container", "delete"]
    if re.search(r"\bkubectl\s+cp\b", command):
        return ["manage_container", "transfer"]
    if re.search(r"\bkubectl\s+port-forward\b", command):
        return ["manage_container", "connect"]
    if re.search(r"\bterraform\s+destroy\b", command):
        return ["manage_cloud", "delete"]
    if re.search(r"\bterraform\s+(?:apply|import|taint|untaint)\b", command):
        return ["manage_cloud", "modify"]
    if re.search(r"\b(?:aws\s+ec2\s+terminate-instances|gcloud\s+compute\s+instances\s+delete|az\s+vm\s+delete)\b", command):
        return ["manage_cloud", "delete"]
    if re.search(r"\b(?:aws\s+ec2\s+(?:start-instances|stop-instances|reboot-instances)|gcloud\s+compute\s+instances\s+(?:start|stop|reset)|az\s+vm\s+(?:start|stop|restart))\b", command):
        return ["manage_cloud", "modify"]
    if re.search(r"\baws\s+s3\s+sync\b", command):
        return ["manage_cloud", "transfer"]
    if re.search(r"\b(?:python|python3)\s+-m\s+pip\s+install\b", command):
        return ["install"]
    if re.search(r"\b(?:python|python3)\s+-m\s+pip\s+install\b.*--dry-run\b", command):
        return ["install"]
    if re.search(r"^\s*(?:sudo\s+)?chage\s+(?:.*\s)?(?:-l|--list)\b", command):
        return ["inspect"]
    if re.search(r"\bmkfs(?:\.[\w+-]+)?\b", command):
        return ["modify"]
    return None


def resolve_intent(record: dict[str, Any]) -> tuple[list[str], str]:
    e = record.get("enrichment") or {}
    override = command_semantic_override(record)
    if override:
        return override, "command_override"
    operation = e.get("operation")
    detail = e.get("operation_detail")
    command = record.get("command", "")

    if operation == "unknown":
        # An unknown executable is still an execution intent. Do not invent a
        # semantic action we cannot establish from syntax; reserve "unknown"
        # for records with no executable program at all.
        if e.get("program"):
            return ["execute"], "unknown_program_execution"
        return ["unknown"], "unknown"

    # Detail always wins for mixed umbrella operations.
    if detail and detail in DETAIL_INTENTS:
        return DETAIL_INTENTS[detail], "detail"

    if operation == "search_and_execute":
        return ["search", "execute"], "detail"

    if operation == "modify_or_execute":
        # Most entries are resolved by the detail table above.
        # Remaining Git/Docker/Kubernetes/NPM forms are handled here.
        if detail:
            if detail.startswith("git_"):
                return ["version_control"], "detail_prefix"
            if detail.startswith(("docker_", "kubectl_")):
                return ["manage_container"], "detail_prefix"
            if detail in {"npm_install"}:
                return ["install"], "detail_prefix"
            if detail in {"npm_update"}:
                return ["manage_package"], "detail_prefix"
            if detail in {"npm_run", "npm_exec"}:
                return ["execute"], "detail_prefix"

    if operation == "delete_or_modify":
        if detail:
            if detail == "git_reset":
                return ["version_control"], "detail"
            if detail.startswith(("docker_", "kubectl_")):
                return ["manage_container"], "detail_prefix"

    if operation == "network_request":
        if detail in {"http_download", "download"}:
            return ["transfer"], "detail"
        return ["network_request"], "operation"

    if operation == "package_management":
        labels, source = package_detail_intents(detail, command)
        if labels:
            return labels, source

    if operation == "remote_access":
        if detail == "sftp":
            return ["connect", "transfer"], "detail"
        return ["connect"], "operation"

    if operation == "infrastructure_operation":
        if detail and detail in DETAIL_INTENTS:
            return DETAIL_INTENTS[detail], "detail"
        return ["execute"], "operation"

    # Stable operation-level mapping.
    if operation in OPERATION_INTENTS:
        return OPERATION_INTENTS[operation], "operation"

    # Conservative domain fallbacks for operations not explicitly listed.
    domain = e.get("domain")
    if domain == "version_control":
        return ["version_control"], "domain"
    if domain == "container":
        return ["manage_container"], "domain"
    if domain == "kubernetes":
        return ["manage_container"], "domain"
    if domain == "service":
        return ["manage_service"], "domain"
    if domain == "database":
        return ["manage_database"], "domain"
    if domain == "permissions":
        return ["manage_permissions"], "domain"
    if domain == "user_account":
        return ["manage_identity"], "domain"
    if domain == "network":
        if e.get("network_operation"):
            return ["network_request"], "domain"
        return ["connect"], "domain"
    if domain == "package":
        return ["manage_package"], "domain"

    return ["unknown"], "unresolved"


# ---------------------------------------------------------------------------
# Risk audit
# ---------------------------------------------------------------------------

def risk_profile(record: dict[str, Any]) -> dict[str, Any]:
    """
    Audit the V7 risk features without pretending this is the final risk
    labeling policy. The later label.py stage can freeze the exact rules.
    """
    e = record.get("enrichment") or {}
    rf = e.get("risk_features") or {}

    destructive = bool(rf.get("destructive"))
    modifies_data = bool(rf.get("modifies_data"))
    modifies_permissions = bool(rf.get("modifies_permissions"))
    security_sensitive = bool(rf.get("security_sensitive"))
    privileged = bool(rf.get("privileged"))
    force = bool(rf.get("force"))
    recursive = bool(rf.get("recursive"))
    network = bool(rf.get("network_operation"))
    shell_control = bool(rf.get("has_shell_control"))

    score = 0
    reasons: list[str] = []

    if destructive:
        score += 3
        reasons.append("destructive")
    if modifies_permissions:
        score += 3
        reasons.append("modifies_permissions")
    if privileged:
        score += 3
        reasons.append("privileged")
    if security_sensitive:
        score += 2
        reasons.append("security_sensitive")
    if force:
        score += 2
        reasons.append("force")
    if recursive:
        score += 1
        reasons.append("recursive")
    if modifies_data:
        score += 1
        reasons.append("modifies_data")
    if shell_control:
        score += 1
        reasons.append("shell_control")
    if network:
        score += 1
        reasons.append("network_operation")

    if destructive and privileged and (force or recursive):
        level = "critical"
    elif score >= 6:
        level = "high"
    elif score >= 3:
        level = "medium"
    elif score >= 1:
        level = "low"
    else:
        level = "safe"

    return {
        "risk": level,
        "score": score,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [r.get("id") for r in records]
    duplicate_ids = [k for k, v in Counter(ids).items() if v > 1]

    invalid_intents = set()
    resolved = 0
    unknown = 0
    source_counts = Counter()
    intent_counts = Counter()
    risk_counts = Counter()
    ambiguous_ops = Counter()
    unresolved_rows = []

    for r in records:
        labels, source = resolve_intent(r)
        source_counts[source] += 1

        for label in labels:
            intent_counts[label] += 1
            if label not in INTENTS and label != "unknown":
                invalid_intents.add(label)

        if labels == ["unknown"]:
            unknown += 1
            unresolved_rows.append(r.get("id"))
        else:
            resolved += 1

        rp = risk_profile(r)
        risk_counts[rp["risk"]] += 1

        operation = (r.get("enrichment") or {}).get("operation")
        if source in {"unresolved", "unknown"}:
            ambiguous_ops[operation] += 1

    return {
        "records": len(records),
        "duplicate_ids": duplicate_ids,
        "invalid_intents": sorted(invalid_intents),
        "resolved_rows": resolved,
        "unknown_rows": unknown,
        "unresolved_ids": unresolved_rows,
        "source_counts": dict(source_counts),
        "intent_counts": dict(intent_counts),
        "risk_counts": dict(risk_counts),
        "ambiguous_operations": dict(ambiguous_ops),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def audit(input_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = []
    malformed = []

    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                malformed.append({"line": line_no, "error": str(exc)})

    validation = validate_records(records)

    operation_counts = Counter(
        (r.get("enrichment") or {}).get("operation") for r in records
    )
    detail_counts = Counter(
        (r.get("enrichment") or {}).get("operation_detail") for r in records
    )
    domain_counts = Counter(
        (r.get("enrichment") or {}).get("domain") for r in records
    )
    program_counts = Counter(
        (r.get("enrichment") or {}).get("program") for r in records
    )

    operation_mapping = defaultdict(Counter)
    operation_examples = defaultdict(list)

    for r in records:
        e = r.get("enrichment") or {}
        operation = e.get("operation")
        labels, source = resolve_intent(r)
        label_text = "+".join(labels)

        operation_mapping[operation][(label_text, source)] += 1

        if len(operation_examples[operation]) < 3:
            operation_examples[operation].append(r.get("command", ""))

    mixed_operations = [
        "search_and_execute",
        "modify_or_execute",
        "delete_or_modify",
        "network_request",
        "package_management",
        "inspect_system",
        "manage_service",
        "inspect_command",
        "checksum",
        "remote_access",
        "infrastructure_operation",
    ]

    mixed_report = {}
    for operation in mixed_operations:
        rows = []
        for (mapping, source), count in operation_mapping.get(operation, {}).items():
            rows.append({
                "candidate_intent": mapping,
                "resolution": source,
                "count": count,
            })
        mixed_report[operation] = {
            "total": operation_counts.get(operation, 0),
            "mappings": sorted(rows, key=lambda x: (-x["count"], x["candidate_intent"])),
            "examples": operation_examples.get(operation, []),
        }

    report = {
        "schema_version": ONTOLOGY_VERSION,
        "input": str(input_path.resolve()),
        "ontology": {
            "intents": INTENTS,
            "risk_levels": RISK_LEVELS,
            "intent_mode": "multi_label",
            "unknown_training_policy": "exclude_from_supervised_training",
        },
        "dataset": {
            "records": len(records),
            "malformed": malformed,
            "unique_programs": len(program_counts),
            "unique_domains": len(domain_counts),
            "unique_operations": len(operation_counts),
            "unique_operation_details": len(detail_counts),
        },
        "validation": validation,
        "top_programs": program_counts.most_common(30),
        "top_domains": domain_counts.most_common(30),
        "top_operations": operation_counts.most_common(100),
        "mixed_operation_audit": mixed_report,
        "risk_audit": {
            "distribution": validation["risk_counts"],
            "note": "Risk scoring is an audit heuristic only; freeze final risk labeling separately in label.py.",
        },
        "status": (
            "PASS"
            if not malformed
            and not validation["duplicate_ids"]
            and not validation["invalid_intents"]
            and validation["unknown_rows"] == 0
            else "REVIEW"
        ),
    }

    return report, records


def write_text_report(report: dict[str, Any], path: Path) -> None:
    d = report["dataset"]
    v = report["validation"]

    lines = [
        "=" * 72,
        "SafeShell Ontology Audit V4",
        "=" * 72,
        f"Input                   : {report['input']}",
        f"Records                 : {d['records']}",
        f"Malformed               : {len(d['malformed'])}",
        f"Duplicate IDs           : {len(v['duplicate_ids'])}",
        f"Unique programs         : {d['unique_programs']}",
        f"Unique domains          : {d['unique_domains']}",
        f"Unique operations       : {d['unique_operations']}",
        f"Unique operation_details: {d['unique_operation_details']}",
        f"Intent labels           : {len(report['ontology']['intents'])}",
        f"Risk labels             : {len(report['ontology']['risk_levels'])}",
        "Intent mode             : MULTI-LABEL",
        f"Resolved rows           : {v['resolved_rows']}",
        f"Unknown rows            : {v['unknown_rows']}",
        "",
        "Intent distribution (record-level label occurrences):",
    ]

    for label, count in sorted(v["intent_counts"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"  {label:28} {count}")

    lines += [
        "",
        "Risk audit distribution:",
    ]
    for level in RISK_LEVELS:
        lines.append(f"  {level:28} {v['risk_counts'].get(level, 0)}")

    lines += [
        "",
        "Resolution sources:",
    ]
    for source, count in sorted(v["source_counts"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"  {source:28} {count}")

    lines += [
        "",
        "Mixed-operation audit:",
    ]

    for operation, data in report["mixed_operation_audit"].items():
        lines.append("")
        lines.append(f"  {operation} ({data['total']})")
        for m in data["mappings"]:
            lines.append(
                f"    {m['count']:5} -> {m['candidate_intent']:30} [{m['resolution']}]"
            )

    lines += [
        "",
        "Unknown/unresolved IDs:",
        "  " + ", ".join(map(str, v["unresolved_ids"])),
        "",
        f"FINAL STATUS: {report['status']}",
        "=" * 72,
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit SafeShell V7 intent ontology.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    report, _ = audit(input_path)

    json_path = output_dir / "ontology_audit_v4.json"
    text_path = output_dir / "ontology_audit_v4.txt"

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_text_report(report, text_path)

    d = report["dataset"]
    v = report["validation"]

    print("=" * 72)
    print("SafeShell Ontology Audit V4")
    print("=" * 72)
    print(f"Input                    : {input_path}")
    print(f"Records                  : {d['records']}")
    print(f"Malformed                : {len(d['malformed'])}")
    print(f"Duplicate IDs            : {len(v['duplicate_ids'])}")
    print(f"Unique programs          : {d['unique_programs']}")
    print(f"Unique domains           : {d['unique_domains']}")
    print(f"Unique operations        : {d['unique_operations']}")
    print(f"Unique operation_details : {d['unique_operation_details']}")
    print(f"Intent labels            : {len(INTENTS)}")
    print(f"Risk labels              : {len(RISK_LEVELS)}")
    print(f"Intent mode              : MULTI-LABEL")
    print(f"Resolved rows            : {v['resolved_rows']}")
    print(f"Unknown rows             : {v['unknown_rows']}")
    print(f"Validation               : {report['status']}")
    print()
    print("Output:")
    print(f"  {json_path}")
    print(f"  {text_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()