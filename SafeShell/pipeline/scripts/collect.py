import json
import re
from pathlib import Path

import bashlex


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "sources" / "manual" / "commands.txt"
OUTPUT_FILE = BASE_DIR / "data" / "raw" / "commands.jsonl"
ERROR_FILE = BASE_DIR / "data" / "raw" / "parse_errors.jsonl"


# ============================================================
# AST helpers
# ============================================================

def get_parts(node):
    """Return child AST nodes if the node has parts."""
    return getattr(node, "parts", []) or []


def collect_nodes(node, kind):
    """Recursively collect AST nodes of a given kind."""

    result = []

    if getattr(node, "kind", None) == kind:
        result.append(node)

    for part in get_parts(node):
        result.extend(collect_nodes(part, kind))

    return result


def find_all_nodes(parts, kind):
    """Collect nodes of a kind from a list of root AST nodes."""

    result = []

    for part in parts:
        result.extend(collect_nodes(part, kind))

    return result


# ============================================================
# Command extraction
# ============================================================

def extract_commands(parts):
    """
    Extract actual command/program names from command nodes.

    Example:

        cp file.txt backup.txt

    -> ["cp"]

        cat file.txt | grep error

    -> ["cat", "grep"]
    """

    commands = []

    command_nodes = find_all_nodes(parts, "command")

    for node in command_nodes:

        node_parts = get_parts(node)

        for part in node_parts:

            if getattr(part, "kind", None) == "word":

                word = getattr(part, "word", "")

                if word:
                    commands.append(word)
                    break

    return list(dict.fromkeys(commands))


# ============================================================
# Word extraction
# ============================================================

def extract_word_nodes(parts):
    """Return all word AST nodes."""

    return find_all_nodes(parts, "word")


def extract_words(parts):
    """Return the raw word values from AST word nodes."""

    return [
        getattr(node, "word", "")
        for node in extract_word_nodes(parts)
        if getattr(node, "word", "")
    ]


# ============================================================
# Flags and arguments
# ============================================================

def is_flag(word):
    """Determine whether a word looks like a command-line flag."""

    return (
        word.startswith("-")
        and word != "-"
    )


def extract_flags(parts):
    """
    Extract flag-like words from the AST.

    Example:

        ls -lah /tmp

    -> ["-lah"]
    """

    flags = []

    for word in extract_words(parts):

        if is_flag(word):
            flags.append(word)

    return list(dict.fromkeys(flags))


def extract_arguments(parts):
    """
    Extract non-command, non-flag words.

    This is a first-level argument extractor.
    It intentionally keeps command-specific semantics for later.

    Example:

        cp file.txt backup.txt

    -> ["file.txt", "backup.txt"]
    """

    arguments = []

    command_nodes = find_all_nodes(parts, "command")

    for command_node in command_nodes:

        node_parts = get_parts(command_node)

        command_seen = False

        for part in node_parts:

            kind = getattr(part, "kind", None)

            if kind == "word":

                word = getattr(part, "word", "")

                if not command_seen:
                    command_seen = True
                    continue

                if not word:
                    continue

                if is_flag(word):
                    continue

                arguments.append(word)

    return arguments


# ============================================================
# Operators
# ============================================================

def extract_operators(command):
    """
    Extract shell operators from the raw command.

    We use lexical scanning while protecting quoted strings.
    """

    operators = [
        "|&",
        "&&",
        "||",
        ";;",
        "&>",
        "&>>",
        ">>",
        "<<<",
        "<<",
        "2>&1",
        "2>>",
        "2>",
        "|",
        ";",
        "&",
        ">",
        "<",
    ]

    found = []

    single_quote = False
    double_quote = False
    escaped = False

    i = 0

    while i < len(command):

        char = command[i]

        if escaped:
            escaped = False
            i += 1
            continue

        if char == "\\" and not single_quote:
            escaped = True
            i += 1
            continue

        if char == "'" and not double_quote:
            single_quote = not single_quote
            i += 1
            continue

        if char == '"' and not single_quote:
            double_quote = not double_quote
            i += 1
            continue

        if not single_quote and not double_quote:

            matched = False

            for operator in operators:

                if command.startswith(operator, i):

                    found.append(operator)

                    i += len(operator)

                    matched = True
                    break

            if matched:
                continue

        i += 1

    return list(dict.fromkeys(found))


# ============================================================
# Redirections
# ============================================================

def extract_redirections(command):
    """Extract shell redirection operators outside quotes."""

    redirections = [
        "2>&1",
        "2>>",
        "2>",
        "<<<",
        "<<",
        ">>",
        "&>>",
        "&>",
        ">",
        "<",
    ]

    found = []

    single_quote = False
    double_quote = False
    escaped = False

    i = 0

    while i < len(command):

        char = command[i]

        if escaped:
            escaped = False
            i += 1
            continue

        if char == "\\" and not single_quote:
            escaped = True
            i += 1
            continue

        if char == "'" and not double_quote:
            single_quote = not single_quote
            i += 1
            continue

        if char == '"' and not single_quote:
            double_quote = not double_quote
            i += 1
            continue

        if not single_quote and not double_quote:

            matched = False

            for redirection in redirections:

                if command.startswith(redirection, i):

                    found.append(redirection)

                    i += len(redirection)

                    matched = True
                    break

            if matched:
                continue

        i += 1

    return list(dict.fromkeys(found))


# ============================================================
# Paths
# ============================================================

def looks_like_path(word):
    """Conservative path detection."""

    if not word:
        return False

    if word.startswith((
        "/",
        "./",
        "../",
        "~/",
    )):
        return True

    if "/" in word:
        return True

    return False


def extract_paths(parts):
    """Extract words that look like filesystem paths."""

    paths = []

    for word in extract_words(parts):

        if looks_like_path(word):
            paths.append(word)

    return list(dict.fromkeys(paths))


# ============================================================
# Environment variables
# ============================================================

def extract_environment_variables(command):
    """
    Extract simple shell variable references.

    Examples:
        $HOME
        $PATH
        ${USER}
    """

    pattern = (
        r"\$(?:"
        r"\{([A-Za-z_][A-Za-z0-9_]*)\}"
        r"|"
        r"([A-Za-z_][A-Za-z0-9_]*)"
        r")"
    )

    matches = re.findall(pattern, command)

    variables = []

    for first, second in matches:

        value = first or second

        if value:
            variables.append(value)

    return list(dict.fromkeys(variables))


# ============================================================
# Detection helpers
# ============================================================

def detect_sudo(parts):
    """Detect sudo as an actual command name."""

    return "sudo" in extract_commands(parts)


def detect_glob(command):
    """Detect common glob characters."""

    return any(
        character in command
        for character in ["*", "?", "["]
    )


def detect_quotes(command):
    """Detect single or double quotes."""

    return "'" in command or '"' in command


def detect_variable_assignment(command):
    """Detect shell variable assignments."""

    pattern = r"(?:^|\s)[A-Za-z_][A-Za-z0-9_]*="

    return bool(
        re.search(pattern, command)
    )


def detect_execution_mode(command):
    """Detect foreground/background execution."""

    stripped = command.strip()

    if stripped.endswith("&") and not stripped.endswith("&&"):
        return "background"

    return "foreground"


def detect_command_substitution(command):
    """Detect command substitution."""

    return (
        "$(" in command
        or "`" in command
    )


def detect_shell_expansion(command):
    """Detect common shell expansion syntax."""

    return any(
        token in command
        for token in [
            "$",
            "~",
            "*",
            "?",
            "{",
            "}",
        ]
    )


# ============================================================
# AST-based structural detection
# ============================================================

def has_kind(parts, kind):
    """Check whether a specific AST node type exists."""

    return len(
        find_all_nodes(parts, kind)
    ) > 0


def detect_pipe(parts):
    """
    Detect an actual pipeline using Bashlex AST.
    """

    return has_kind(parts, "pipeline")


def detect_subshell(parts):
    """
    Detect actual subshell AST nodes.

    This avoids confusing:
        $(...)
        $((...))
        (...)

    with one another.
    """

    return has_kind(parts, "compound")


# ============================================================
# Category detection
# ============================================================

def determine_category(commands):
    """Initial command category classifier."""

    command_names = set(commands)

    if command_names & {
        "ls",
        "cd",
        "pwd",
        "cp",
        "mv",
        "rm",
        "mkdir",
        "rmdir",
        "touch",
        "find",
        "ln",
        "stat",
        "file",
    }:
        return "filesystem"

    if command_names & {
        "chmod",
        "chown",
        "chgrp",
        "setfacl",
        "getfacl",
        "getcap",
        "setcap",
    }:
        return "permissions"

    if command_names & {
        "ps",
        "top",
        "htop",
        "kill",
        "pkill",
        "killall",
        "pgrep",
        "pidof",
    }:
        return "process"

    if command_names & {
        "systemctl",
        "service",
        "journalctl",
    }:
        return "service"

    if command_names & {
        "apt",
        "apt-get",
        "dpkg",
        "pip",
        "pip3",
        "npm",
        "yarn",
        "cargo",
    }:
        return "package"

    if "git" in command_names:
        return "git"

    if command_names & {
        "curl",
        "wget",
        "ssh",
        "scp",
        "rsync",
        "ping",
        "ip",
        "ss",
        "netstat",
        "nc",
        "dig",
        "nslookup",
        "traceroute",
    }:
        return "network"

    if command_names & {
        "df",
        "du",
        "mount",
        "umount",
        "lsblk",
        "blkid",
        "fdisk",
        "parted",
    }:
        return "disk_storage"

    if command_names & {
        "tar",
        "gzip",
        "gunzip",
        "zip",
        "unzip",
        "xz",
        "bzip2",
    }:
        return "archive_compression"

    if command_names & {
        "docker",
        "podman",
        "kubectl",
        "helm",
    }:
        return "container"

    if command_names & {
        "grep",
        "sed",
        "awk",
        "cut",
        "sort",
        "uniq",
        "tr",
        "head",
        "tail",
        "wc",
    }:
        return "text_processing"

    if command_names & {
        "gcc",
        "g++",
        "clang",
        "clang++",
        "make",
        "cmake",
        "python",
        "python3",
        "node",
        "npm",
        "go",
        "cargo",
        "mvn",
        "gradle",
    }:
        return "development"

    if command_names & {
        "useradd",
        "usermod",
        "userdel",
        "passwd",
        "groupadd",
        "groupdel",
        "groupmod",
        "groups",
        "whoami",
        "id",
    }:
        return "user_account"

    if command_names & {
        "iptables",
        "nft",
        "ufw",
        "firewall-cmd",
        "openssl",
        "gpg",
    }:
        return "security"

    return "other"


# ============================================================
# Parse one command
# ============================================================

def parse_command(command, record_id):

    parts = bashlex.parse(command)

    commands = extract_commands(parts)

    flags = extract_flags(parts)

    arguments = extract_arguments(parts)

    operators = extract_operators(command)

    redirections = extract_redirections(command)

    paths = extract_paths(parts)

    environment_variables = (
        extract_environment_variables(command)
    )

    category = determine_category(commands)

    record = {

        "id": record_id,

        "command": command,

        "category": category,

        "subcategory": "unknown",

        "commands": commands,

        "flags": flags,

        "arguments": arguments,

        "operators": operators,

        "redirections": redirections,

        "has_sudo": detect_sudo(parts),

        "has_pipe": detect_pipe(parts),

        "has_redirection": (
            len(redirections) > 0
        ),

        "has_chaining": any(
            operator in operators
            for operator in [
                "&&",
                "||",
                ";",
                ";;",
            ]
        ),

        "has_command_substitution": (
            detect_command_substitution(command)
        ),

        "has_variable_assignment": (
            detect_variable_assignment(command)
        ),

        "has_glob": (
            detect_glob(command)
        ),

        "has_shell_expansion": (
            detect_shell_expansion(command)
        ),

        "has_quotes": (
            detect_quotes(command)
        ),

        "has_subshell": (
            detect_subshell(parts)
        ),

        "execution_mode": (
            detect_execution_mode(command)
        ),

        "target_types": [],

        "paths": paths,

        "environment_variables": (
            environment_variables
        ),

        "description": "",

        "source": {
            "type": "manual",
            "name": "SafeShell Team",
            "reference": "",
            "license": ""
        },

        "tags": []
    }

    return record


# ============================================================
# Main collection function
# ============================================================

def collect_commands():

    if not INPUT_FILE.exists():

        print(
            f"Input file not found: {INPUT_FILE}"
        )

        return

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    ERROR_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    records = []

    errors = []

    with INPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        record_id = 1

        for line_number, line in enumerate(
            file,
            start=1
        ):

            command = line.strip()

            if not command:
                continue

            try:

                record = parse_command(
                    command,
                    record_id
                )

                records.append(record)

                record_id += 1

            except Exception as error:

                error_record = {
                    "line": line_number,
                    "command": command,
                    "error_type": type(error).__name__,
                    "error": str(error)
                }

                errors.append(
                    error_record
                )

                print(
                    f"Could not parse line "
                    f"{line_number}: {command}"
                )

                print(
                    f"Error: {error}"
                )

    # --------------------------------------------------------
    # Write successful records
    # --------------------------------------------------------

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        for record in records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

    # --------------------------------------------------------
    # Write parse errors
    # --------------------------------------------------------

    with ERROR_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        for error in errors:

            file.write(
                json.dumps(
                    error,
                    ensure_ascii=False
                )
                + "\n"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("SafeShell Collection Complete")
    print("=" * 60)

    print(
        f"Successfully parsed : {len(records)}"
    )

    print(
        f"Parse failures       : {len(errors)}"
    )

    print(
        f"Total processed      : "
        f"{len(records) + len(errors)}"
    )

    print()
    print(
        f"Dataset written to   : {OUTPUT_FILE}"
    )

    print(
        f"Errors written to    : {ERROR_FILE}"
    )


# ============================================================
# Program entry point
# ============================================================

if __name__ == "__main__":
    collect_commands()