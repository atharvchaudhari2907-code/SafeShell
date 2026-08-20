from __future__ import annotations

import json
import re
import shlex
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "raw" / "commands.jsonl"
OUTPUT_FILE = BASE_DIR / "data" / "enriched" / "enriched_commands_v11.jsonl"

SCHEMA_VERSION = "11.0"

SHELL_BUILTINS = {
    "alias", "bg", "bind", "break", "builtin", "caller", "cd", "command", "compgen",
    "complete", "compopt", "continue", "declare", "dirs", "disown", "echo", "enable",
    "eval", "exec", "exit", "export", "false", "fc", "fg", "getopts", "hash", "help",
    "history", "jobs", "kill", "let", "local", "logout", "mapfile", "popd", "printf",
    "pushd", "pwd", "read", "readonly", "return", "set", "shift", "shopt", "source",
    "suspend", "test", "times", "trap", "true", "type", "typeset", "ulimit", "umask",
    "unalias", "unset", "wait", "."
}

WRAPPERS = {"sudo", "doas", "env", "nohup", "timeout", "nice", "ionice", "chrt", "taskset", "stdbuf", "flock", "xargs"}
WRAPPER_VALUE_OPTIONS = {
    "sudo": {"-u", "-g", "-h", "-p"}, "doas": {"-u"},
    "timeout": {"-k", "--kill-after"}, "nice": {"-n"},
    "ionice": {"-c", "-n", "-p"}, "chrt": {"-p"}, "taskset": {"-c", "-p"},
    "stdbuf": {"-i", "-o", "-e"}, "flock": {"-w", "-E", "-o", "-c"},
    "xargs": {"-n", "-s", "-P", "-L", "-I", "-d", "-a"},
}

DIRECT_OPERATIONS = {
    "pwd": ("shell", "inspect_location", "print_current_directory"),
    "declare": ("shell", "inspect_environment", "declare"),
    "readonly": ("shell", "shell_configuration", "readonly"),
    "bash": ("shell", "execute_program", "shell_interpreter"),
    "sh": ("shell", "execute_program", "shell_interpreter"),
    "zsh": ("shell", "execute_program", "shell_interpreter"),
    "dash": ("shell", "execute_program", "shell_interpreter"),
    "lastb": ("system", "inspect_login_history", "failed_login_history"),
    "cd": ("filesystem", "change_directory", "change_working_directory"),
    "ls": ("filesystem", "list", "list_directory"), "dir": ("filesystem", "list", "list_directory"),
    "tree": ("filesystem", "list", "list_tree"), "cat": ("filesystem", "read", "file_read"),
    "tac": ("filesystem", "read", "reverse_file_read"), "head": ("filesystem", "read", "read_head"),
    "tail": ("filesystem", "read", "read_tail"), "less": ("filesystem", "read", "interactive_file_read"),
    "more": ("filesystem", "read", "interactive_file_read"), "touch": ("filesystem", "create", "create_file"),
    "mkdir": ("filesystem", "create", "create_directory"), "rmdir": ("filesystem", "delete", "remove_directory"),
    "rm": ("filesystem", "delete", "remove_path"), "cp": ("filesystem", "copy", "copy_path"),
    "mv": ("filesystem", "move", "move_path"), "ln": ("filesystem", "link", "create_link"),
    "find": ("filesystem", "search", "filesystem_search"), "locate": ("filesystem", "search", "filesystem_search"),
    "realpath": ("filesystem", "inspect", "resolve_path"), "readlink": ("filesystem", "inspect", "resolve_link"),
    "basename": ("filesystem", "inspect", "basename"), "dirname": ("filesystem", "inspect", "dirname"),
    "file": ("filesystem", "inspect", "file_type"), "stat": ("filesystem", "inspect", "file_metadata"),
    "du": ("disk_storage", "inspect_usage", "directory_usage"), "df": ("disk_storage", "inspect_usage", "filesystem_usage"),
    "lsblk": ("disk_storage", "inspect_devices", "block_devices"), "findmnt": ("disk_storage", "inspect_mounts", "mount_information"),
    "mount": ("disk_storage", "mount", "mount_filesystem"), "umount": ("disk_storage", "unmount", "unmount_filesystem"),
    "fstrim": ("disk_storage", "trim", "discard_unused_blocks"),
    "chmod": ("permissions", "change_permissions", "chmod"), "chown": ("permissions", "change_owner", "chown"),
    "chgrp": ("permissions", "change_group", "chgrp"), "setfacl": ("permissions", "change_permissions", "acl"),
    "setcap": ("permissions", "change_capabilities", "file_capabilities"), "getfacl": ("permissions", "inspect_permissions", "acl"),
    "getcap": ("permissions", "inspect_capabilities", "file_capabilities"), "umask": ("permissions", "configure", "umask"),
    "ps": ("process", "inspect_processes", "process_list"), "top": ("process", "inspect_processes", "interactive_process_monitor"),
    "htop": ("process", "inspect_processes", "interactive_process_monitor"), "pgrep": ("process", "search_processes", "process_search"),
    "pidof": ("process", "inspect_processes", "process_pid_lookup"), "lsof": ("process", "inspect_resources", "open_files"),
    "fuser": ("process", "inspect_resources", "resource_users"), "kill": ("process", "terminate_process", "signal_process"),
    "pkill": ("process", "terminate_process", "signal_processes"), "killall": ("process", "terminate_process", "signal_processes"),
    "jobs": ("process", "inspect_jobs", "shell_jobs"), "fg": ("process", "resume_job", "foreground_job"),
    "bg": ("process", "resume_job", "background_job"), "wait": ("process", "wait", "wait_for_process"),
    "disown": ("process", "detach_job", "detach_shell_job"),
    "systemctl": ("service", "manage_service", "service_manager"), "service": ("service", "manage_service", "service_manager"),
    "env": ("shell", "inspect_environment", "env"), "xargs": ("shell", "execute_arguments", "xargs"),
    "sudo": ("security", "privilege_wrapper", "sudo"), "doas": ("security", "privilege_wrapper", "doas"),
    "command": ("shell", "inspect_command", "command_builtin"),
    "terraform": ("development", "infrastructure_operation", "terraform"),
    "aws": ("cloud", "cloud_operation", "aws"), "gcloud": ("cloud", "cloud_operation", "gcloud"), "az": ("cloud", "cloud_operation", "azure_cli"),
    "loginctl": ("service", "manage_service", "loginctl"), "hostnamectl": ("system", "inspect_system", "hostnamectl"),
    "lscpu": ("system", "inspect_resources", "cpu_information"), "ssh-add": ("security", "key_management", "ssh_agent"),
    "nft": ("security", "firewall_management", "nftables"), "ausearch": ("security", "audit_search", "audit_logs"),
    "resolvectl": ("network", "dns_configuration", "systemd_resolved"), "trap": ("shell", "signal_handler", "trap"),
    "gradle": ("development", "build", "gradle"), "dotnet": ("development", "build", "dotnet"),
    "redis-cli": ("database", "database_operation", "redis"),
    "[": ("shell", "conditional_test", "test_builtin"), "type": ("shell", "inspect_command", "type_builtin"),
    "which": ("shell", "inspect_command", "command_lookup"), "alias": ("shell", "shell_configuration", "alias"),
    "sleep": ("system", "wait", "sleep"), "ssh-keyscan": ("security", "key_management", "ssh_host_key_scan"),
    "auditctl": ("security", "audit_management", "auditctl"), "setenforce": ("security", "security_configuration", "selinux_mode"),
    "networkctl": ("network", "network_configuration", "systemd_networkctl"), "systemd-analyze": ("system", "inspect_system", "systemd_analysis"),
    "timedatectl": ("system", "system_configuration", "time_configuration"), "lsmem": ("system", "inspect_resources", "memory_layout"),
    "lsusb": ("system", "inspect_devices", "usb_devices"), "lspci": ("system", "inspect_devices", "pci_devices"),
    "black": ("development", "format_code", "black"), "flake8": ("development", "lint_code", "flake8"), "mypy": ("development", "type_check", "mypy"),
    "install": ("filesystem", "install", "install_program"), "paste": ("text_processing", "combine_columns", "paste"),
    "join": ("text_processing", "join_records", "join"), "split": ("filesystem", "split_file", "split"),
    "cmp": ("text_processing", "compare", "cmp"), "blkid": ("disk_storage", "inspect_devices", "filesystem_identity"),
    "sync": ("disk_storage", "sync", "flush_filesystem_buffers"), "users": ("user_account", "inspect_users", "logged_in_users"),
    "renice": ("process", "change_priority", "renice"), "whereis": ("shell", "inspect_command", "whereis"),
    "unalias": ("shell", "shell_configuration", "unalias"), "tracepath": ("network", "route_diagnostic", "tracepath"),
    "swapon": ("disk_storage", "swap_management", "swapon"), "atq": ("automation", "inspect_scheduled_tasks", "atq"),
    "atrm": ("automation", "delete_scheduled_task", "atrm"), "ctest": ("development", "test", "ctest"),
    "configure": ("development", "configure_build", "configure"), "auditd": ("security", "audit_management", "auditd"),
    "semanage": ("security", "security_configuration", "selinux_policy"), "getenforce": ("security", "security_configuration", "selinux_mode"),
    "aa-status": ("security", "security_status", "apparmor"), "apparmor_parser": ("security", "security_configuration", "apparmor_parser"),
    "nmtui": ("network", "network_configuration", "network_manager_tui"), "sudoedit": ("security", "privileged_edit", "sudoedit"),
    "visudo": ("security", "privileged_edit", "sudoers_validation"), "faillock": ("security", "account_lockout", "faillock"),
    "capsh": ("security", "capability_inspection", "capsh"), "ip6tables": ("security", "firewall_management", "ip6tables"),
    "aureport": ("security", "audit_report", "aureport"), "sestatus": ("security", "security_status", "selinux_status"),
    "apparmor_status": ("security", "security_status", "apparmor"), "partprobe": ("disk_storage", "partition", "partprobe"),
    "tune2fs": ("disk_storage", "filesystem_configuration", "tune2fs"), "resize2fs": ("disk_storage", "filesystem_configuration", "resize2fs"),
    "pvdisplay": ("disk_storage", "inspect_storage", "lvm_physical_volume"), "vgdisplay": ("disk_storage", "inspect_storage", "lvm_volume_group"),
    "lvdisplay": ("disk_storage", "inspect_storage", "lvm_logical_volume"), "pvs": ("disk_storage", "inspect_storage", "lvm_physical_volumes"),
    "vgs": ("disk_storage", "inspect_storage", "lvm_volume_groups"), "lvs": ("disk_storage", "inspect_storage", "lvm_logical_volumes"),
    "lvscan": ("disk_storage", "inspect_storage", "lvm_scan"), "dmsetup": ("disk_storage", "device_mapper", "dmsetup"),
    "builtin": ("shell", "shell_configuration", "builtin"), "mapfile": ("shell", "read_input", "mapfile"),
    "ansible-inventory": ("automation", "inspect_inventory", "ansible_inventory"), "ruby": ("development", "execute_program", "ruby"),
    "php": ("development", "execute_program", "php"), "snap": ("package", "package_management", "snap"), "flatpak": ("package", "package_management", "flatpak"),
    "sftp": ("network", "remote_access", "sftp"),
    "route": ("network", "network_configuration", "route"), "arp": ("network", "network_configuration", "arp"),
    "socat": ("network", "network_connection", "socat"), "netcat": ("network", "network_connection", "netcat"),
    "history": ("shell", "shell_history", "history"), "shift": ("shell", "shell_parameter", "shift"),
    "getopts": ("shell", "parse_options", "getopts"), "return": ("shell", "function_control", "return"),
    "exit": ("shell", "shell_control", "exit"),

    "journalctl": ("system_logging", "inspect_logs", "journal_logs"), "dmesg": ("system_logging", "inspect_logs", "kernel_logs"),
    "logger": ("system_logging", "write_log", "system_log_write"),
    "apt": ("package", "package_management", "apt"), "apt-get": ("package", "package_management", "apt_get"),
    "apt-cache": ("package", "inspect_package", "apt_cache"), "apt-mark": ("package", "manage_package_state", "apt_mark"),
    "dpkg": ("package", "package_management", "dpkg"), "dpkg-query": ("package", "inspect_package", "dpkg_query"),
    "pip": ("package", "package_management", "pip"), "pip3": ("package", "package_management", "pip"),
    "npm": ("package", "package_management", "npm"), "npx": ("development", "execute_package_binary", "npx"),
    "git": ("version_control", "version_control", "git"),
    "curl": ("network", "network_request", "http_request"), "wget": ("network", "network_request", "download"),
    "ssh": ("network", "remote_access", "secure_shell"), "scp": ("network", "file_transfer", "secure_copy"),
    "rsync": ("network", "file_transfer", "synchronization"), "ping": ("network", "connectivity_test", "icmp_probe"),
    "nc": ("network", "network_connection", "netcat"), "dig": ("network", "dns_lookup", "dns_query"),
    "host": ("network", "dns_lookup", "dns_query"), "nslookup": ("network", "dns_lookup", "dns_query"),
    "ip": ("network", "network_configuration", "ip_tool"), "ss": ("network", "inspect_connections", "socket_statistics"),
    "netstat": ("network", "inspect_connections", "network_statistics"), "traceroute": ("network", "route_diagnostic", "traceroute"),
    "mtr": ("network", "route_diagnostic", "mtr"), "ethtool": ("network", "inspect_interface", "ethernet_tool"),
    "nmcli": ("network", "network_configuration", "network_manager_cli"), "ufw": ("security", "firewall_management", "ufw"),
    "firewall-cmd": ("security", "firewall_management", "firewalld"), "iptables": ("security", "firewall_management", "iptables"),
    "tcpdump": ("network", "packet_capture", "tcpdump"),
    "tar": ("archive_compression", "archive", "tar"), "gzip": ("archive_compression", "compress", "gzip"),
    "gunzip": ("archive_compression", "decompress", "gzip"), "bzip2": ("archive_compression", "compress", "bzip2"),
    "bunzip2": ("archive_compression", "decompress", "bzip2"), "xz": ("archive_compression", "compress", "xz"),
    "unxz": ("archive_compression", "decompress", "xz"), "zstd": ("archive_compression", "compress", "zstd"),
    "zip": ("archive_compression", "archive", "zip"), "unzip": ("archive_compression", "extract", "unzip"), "7z": ("archive_compression", "archive", "7zip"),
    "grep": ("text_processing", "search_text", "grep"), "egrep": ("text_processing", "search_text", "grep"),
    "fgrep": ("text_processing", "search_text", "grep"), "sed": ("text_processing", "transform_text", "sed"),
    "awk": ("text_processing", "extract_fields", "awk"), "cut": ("text_processing", "extract_fields", "cut"),
    "sort": ("text_processing", "sort", "sort"), "uniq": ("text_processing", "deduplicate", "uniq"),
    "tr": ("text_processing", "transform_text", "tr"), "wc": ("text_processing", "count", "word_count"),
    "tee": ("text_processing", "split_output", "tee"), "diff": ("text_processing", "compare", "diff"),
    "comm": ("text_processing", "compare", "comm"), "echo": ("shell", "output", "echo"),
    "printf": ("shell", "formatted_output", "printf"), "export": ("shell", "set_environment", "export"),
    "unset": ("shell", "unset_environment", "unset"), "set": ("shell", "shell_configuration", "set"),
    "shopt": ("shell", "shell_configuration", "shopt"), "source": ("shell", "source_script", "source"),
    ".": ("shell", "source_script", "source"), "exec": ("shell", "replace_process", "exec"),
    "eval": ("shell", "evaluate_command", "eval"), "read": ("shell", "read_input", "read"),
    "test": ("shell", "conditional_test", "test"), "true": ("shell", "return_status", "true"), "false": ("shell", "return_status", "false"),
    "python": ("development", "execute_program", "python"), "python3": ("development", "execute_program", "python"),
    "node": ("development", "execute_program", "node"), "java": ("development", "execute_program", "java"),
    "gcc": ("development", "compile", "gcc"), "g++": ("development", "compile", "g++"),
    "clang": ("development", "compile", "clang"), "clang++": ("development", "compile", "clang++"),
    "make": ("development", "build", "make"), "cmake": ("development", "build", "cmake"),
    "pytest": ("development", "test", "pytest"), "cargo": ("development", "build", "cargo"), "go": ("development", "build", "go"),
    "mvn": ("development", "build", "maven"), "sqlite3": ("database", "database_operation", "sqlite3"),
    "psql": ("database", "database_operation", "postgresql"), "mysql": ("database", "database_operation", "mysql"),
    "pg_dump": ("database", "database_backup", "postgres_dump"), "pg_restore": ("database", "database_restore", "postgres_restore"),
    "mysqldump": ("database", "database_backup", "mysql_dump"), "mongosh": ("database", "database_operation", "mongodb"),
    "openssl": ("security", "cryptographic_operation", "openssl"), "gpg": ("security", "cryptographic_operation", "gpg"),
    "ssh-keygen": ("security", "key_management", "ssh_keygen"), "sha256sum": ("security", "checksum", "sha256"),
    "sha512sum": ("security", "checksum", "sha512"), "sha1sum": ("security", "checksum", "sha1"),
    "md5sum": ("security", "checksum", "md5"), "cksum": ("security", "checksum", "cksum"),
    "getent": ("system", "identity_lookup", "getent"), "hostname": ("system", "inspect_system", "hostname"),
    "uname": ("system", "inspect_system", "uname"), "uptime": ("system", "inspect_system", "uptime"),
    "free": ("system", "inspect_resources", "memory"), "iostat": ("system", "inspect_resources", "io_statistics"),
    "vmstat": ("system", "inspect_resources", "virtual_memory_statistics"), "mpstat": ("system", "inspect_resources", "cpu_statistics"),
    "date": ("system", "inspect_system", "date"), "whoami": ("user_account", "inspect_identity", "whoami"),
    "id": ("user_account", "inspect_identity", "id"), "groups": ("user_account", "inspect_groups", "groups"),
    "who": ("user_account", "inspect_sessions", "who"), "w": ("user_account", "inspect_sessions", "w"),
    "last": ("user_account", "inspect_login_history", "last"), "lastlog": ("user_account", "inspect_login_history", "lastlog"),
    "useradd": ("user_account", "manage_user", "create_user"), "usermod": ("user_account", "manage_user", "modify_user"),
    "userdel": ("user_account", "manage_user", "delete_user"), "groupadd": ("user_account", "manage_group", "create_group"),
    "groupdel": ("user_account", "manage_group", "delete_group"), "gpasswd": ("user_account", "manage_group", "group_password"),
    "passwd": ("user_account", "manage_user", "password_change"), "chage": ("user_account", "manage_user", "account_aging"),
    "reboot": ("system", "system_control", "reboot"), "shutdown": ("system", "system_control", "shutdown"),
    "poweroff": ("system", "system_control", "poweroff"), "dd": ("disk_storage", "raw_io", "block_copy"),
    "fdisk": ("disk_storage", "partition", "fdisk"), "parted": ("disk_storage", "partition", "parted"),
    "fsck": ("disk_storage", "filesystem_check", "fsck"), "e2fsck": ("disk_storage", "filesystem_check", "e2fsck"),
    "mkfs.ext4": ("disk_storage", "format", "filesystem_format"),
    "docker": ("container", "container_operation", "docker"), "kubectl": ("kubernetes", "kubernetes_operation", "kubectl"),
    "helm": ("kubernetes", "kubernetes_operation", "helm"), "ansible": ("automation", "configuration_automation", "ansible"),
    "ansible-playbook": ("automation", "configuration_automation", "ansible_playbook"), "crontab": ("automation", "schedule_task", "cron"),
    "at": ("automation", "schedule_task", "at"), "watch": ("automation", "repeat_command", "watch"),
    "printenv": ("shell", "inspect_environment", "printenv"), "seq": ("text_processing", "generate_sequence", "seq"),
    "jq": ("text_processing", "transform_json", "jq"), "namei": ("filesystem", "inspect", "namei"),
}

SUBCOMMANDS = {
    "git": {"add","am","archive","bisect","branch","bundle","cat-file","checkout","cherry-pick","clean","clone","commit","config","describe","diff","fetch","format-patch","gc","grep","init","log","ls-files","merge","mv","pull","push","rebase","reflog","remote","reset","restore","revert","rev-parse","rm","show","shortlog","stash","status","switch","tag","worktree"},
    "docker": {"attach","build","commit","compose","cp","create","diff","exec","export","history","image","images","import","info","inspect","kill","load","login","logout","logs","network","pause","plugin","port","ps","pull","push","rename","restart","rm","rmi","run","save","search","start","stats","stop","system","tag","top","unpause","update","version","volume"},
    "kubectl": {"apply","attach","auth","autoscale","config","cp","create","delete","describe","diff","edit","exec","explain","expose","get","logs","patch","port-forward","proxy","replace","rollout","run","scale","set","taint","top","wait","version","cluster-info","label","annotate"},
    "helm": {"create","dependency","env","get","history","install","lint","list","package","plugin","pull","repo","rollback","search","show","status","template","test","uninstall","upgrade","version"},
    "npm": {"access","audit","cache","ci","config","dedupe","doctor","exec","fund","init","install","link","login","outdated","pack","publish","run","search","test","uninstall","update","view","version"},
    "pip": {"install","download","uninstall","freeze","list","show","check","config","cache","wheel","index","debug"},
    "apt": {"update","upgrade","full-upgrade","install","remove","purge","autoremove","search","show","list","edit-sources","satisfy"},
    "apt-get": {"update","upgrade","dist-upgrade","install","remove","purge","autoremove","source","build-dep","download"},
    "systemctl": {"start","stop","restart","reload","try-restart","reload-or-restart","enable","disable","mask","unmask","status","is-active","is-enabled","is-failed","list-units","list-unit-files","list-dependencies","daemon-reload","reset-failed","kill","show","cat","edit","set-property","get-default","set-default","is-system-running","reboot","poweroff","halt","suspend","hibernate","list-timers"},
    "terraform": {"apply","destroy","fmt","get","graph","import","init","output","plan","providers","refresh","show","state","taint","test","untaint","validate","version","workspace"},
    "aws": {"configure","ec2","s3","iam","lambda","cloudformation","logs","sts","eks","ecr","route53","rds","dynamodb","ssm","secretsmanager","kms","cloudwatch","autoscaling","elb","sns","sqs","codebuild","codecommit","codepipeline"},
    "gcloud": {"auth","config","compute","container","storage","projects","iam","functions","run","sql","dns","logging","pubsub","artifacts","app","components","info","version"},
    "az": {"login","account","group","vm","storage","network","webapp","aks","acr","functionapp","sql","keyvault","role","resource","monitor","container","disk"},
    "git-lfs": {"install","track","untrack","pull","push","fetch","status","ls-files","prune","env","logs"},
}

PROGRAM_FLAG_RISK_MAP = {
    "rm": {"recursive": {"-r","-R","--recursive","-rf","-fr","-Rrf","-Rf"}, "force": {"-f","-rf","-fr","-Rrf","-Rf"}},
    "cp": {"recursive": {"-r","-R","--recursive","-a","-ar","-ra"}, "force": {"-f","--force"}},
    "mv": {"force": {"-f","--force"}},
    "chmod": {"recursive": {"-R","--recursive"}},
    "chown": {"recursive": {"-R","--recursive"}},
    "chgrp": {"recursive": {"-R","--recursive"}},
    "setfacl": {"recursive": {"-R","--recursive"}},
    "rsync": {"recursive": {"-r","-a","--recursive"}, "force": {"--force"}},
    "git": {"force": {"-f","--force","--hard","-fd","-fdx","-df","-dfx"}},
    "docker": {"force": {"-f","--force"}},
    "kubectl": {"force": {"--force"}},
    "userdel": {"recursive": {"-r","--remove"}, "force": {"-f","--force"}},
    "groupdel": {"force": {"-f","--force"}},
    "sort": {}, "uname": {}, "cut": {}, "pgrep": {}, "journalctl": {}, "readlink": {},
    "grep": {}, "head": {}, "tail": {}, "kill": {}, "pkill": {}, "killall": {},
}
FLAG_GROUPS = {"privileged": {"--privileged"}}

DANGEROUS_PROGRAMS = {"rm","rmdir","shred","mkfs.ext4","mkfs","dd","fdisk","parted","reboot","shutdown","poweroff","userdel","groupdel","kill","pkill","killall","chmod","chown","setfacl","setcap","systemctl","apt","apt-get","dpkg","curl","wget","ssh"}

ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
ENV_RE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")
URL_RE = re.compile(r"^(?:https?|ssh|git|ftp)://", re.I)
PATH_EXT_RE = re.compile(r"\.(?:txt|log|json|yaml|yml|csv|xml|conf|cfg|ini|sh|py|js|ts|cpp|c|h|hpp|md|sql|db|sqlite|tar|gz|zip|img|iso)$", re.I)

# Longest first. Used only outside quotes.
OPERATORS = (";;&", "<<<", "&>>", "2>&1", "2>&-", "&&", "||", "|&", ">>", "<<", "&>", ";;", ";&", "|", ">", "<", "&", ";")


def unique(items: Iterable[Any]) -> list[Any]:
    return list(dict.fromkeys(x for x in items if x is not None and x != ""))


def lexical_words(command: str) -> tuple[list[str], str]:
    try:
        return shlex.split(command, posix=True), "shlex"
    except (ValueError, TypeError):
        return command.split(), "split"


def scan_tokens(command: str) -> tuple[list[str], list[str]]:
    """Lightweight shell scanner used only as fallback; it never claims Bash AST semantics."""
    words: list[str] = []
    operators: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    i = 0

    def flush() -> None:
        if buf:
            words.append("".join(buf))
            buf.clear()

    while i < len(command):
        ch = command[i]
        if quote:
            buf.append(ch)
            if ch == "\\" and quote == '"' and i + 1 < len(command):
                i += 1
                buf.append(command[i])
            elif ch == quote:
                quote = None
            i += 1
            continue
        if ch in "'\"":
            quote = ch
            buf.append(ch)
            i += 1
            continue
        matched = False
        for op in OPERATORS:
            if command.startswith(op, i):
                flush()
                operators.append(op)
                i += len(op)
                matched = True
                break
        if matched:
            continue
        if ch.isspace():
            flush()
        else:
            buf.append(ch)
        i += 1
    flush()
    return words, operators


def split_pipeline(command: str) -> list[str]:
    segments: list[str] = []
    start = 0
    quote: str | None = None
    depth = 0
    i = 0
    while i < len(command):
        ch = command[i]
        if quote:
            if ch == "\\" and quote == '"':
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "'\"":
            quote = ch
            i += 1
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == "|" and depth == 0:
            if i + 1 < len(command) and command[i + 1] in "|&":
                i += 2
                continue
            segments.append(command[start:i].strip())
            i += 1
            start = i
            continue
        i += 1
    segments.append(command[start:].strip())
    return [s for s in segments if s]


def first_executable_word(segment: str) -> str | None:
    words, _ = scan_tokens(segment)
    if not words:
        return None
    # Parenthesized single commands such as `(date)` are still one executable.
    if len(words) == 1 and words[0].startswith("(") and words[0].endswith(")"):
        return words[0].strip("()") or None
    i = 0
    while i < len(words):
        token = words[i]
        if ASSIGN_RE.match(token):
            # A top-level assignment does not itself identify an executable.
            # If the assignment contains command substitution, keep it as a shell assignment.
            if i == 0 and "$(" in token:
                return None
            i += 1
            continue
        if token in {"(", ")", "{", "}", "if", "then", "else", "elif", "do"}:
            i += 1
            continue
        if token.startswith("(") and len(token) > 1:
            token = token.lstrip("(")
        if token.endswith(")") and token.count("(") == 0:
            token = token.rstrip(")")
        # Function definition name, not an executable program.
        if token.endswith("()"):
            return None
        words[i] = token
        break
    while i < len(words) and words[i] in WRAPPERS:
        wrapper = words[i]
        i += 1
        while i < len(words):
            flag = words[i]
            if flag in WRAPPER_VALUE_OPTIONS.get(wrapper, set()):
                i += 2
                continue
            if flag.startswith("--signal=") or flag.startswith("--kill-after="):
                i += 1
                continue
            if flag.startswith("-"):
                i += 1
                continue
            break
        if wrapper == "env":
            while i < len(words) and (ASSIGN_RE.match(words[i]) or words[i].startswith("-")):
                i += 1
        if wrapper == "timeout":
            while i < len(words) and re.fullmatch(r"\d+(?:\.\d+)?[smhd]", words[i]):
                i += 1
        if i >= len(words):
            return wrapper
    if i < len(words) and not words[i].startswith("-"):
        return words[i]
    if words and words[0] in WRAPPERS:
        return words[0]
    return None


def extract_programs(command: str, source_commands: Any) -> tuple[list[str], str]:
    # Function definitions are shell constructs; do not promote their body command
    # to the program for the outer record.
    if re.match(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*\(\)\s*\{", command):
        return [], "function_definition"
    # commands.jsonl was produced by the Bashlex collector. Its `commands` list is
    # therefore the preferred parsed representation for simple/pipeline commands.
    segments = split_pipeline(command)
    programs = [first_executable_word(s) for s in segments]
    programs = unique(p for p in programs if p)
    if programs:
        return programs, "segment_lexical"

    if isinstance(source_commands, list):
        candidates = [
            x for x in source_commands
            if isinstance(x, str) and x and not x.startswith("-")
            and not ASSIGN_RE.match(x)
            and not x.endswith("()")
            and x not in {"(", ")", "{", "}", "if", "then", "else", "elif", "fi", "for", "while", "until", "do", "done", "case", "esac", "in"}
        ]
        if candidates:
            return [candidates[0]], "collector_commands"

    return [], "none"


def extract_flags(words: list[str]) -> list[str]:
    return unique(w for w in words if w.startswith("-") and w != "-")


def extract_assignments(words: list[str]) -> list[str]:
    return unique(w for w in words if ASSIGN_RE.match(w))


def extract_environment_variables(command: str) -> list[str]:
    return unique(a or b for a, b in ENV_RE.findall(command))


def extract_paths(words: list[str]) -> list[str]:
    result: list[str] = []
    for word in words:
        clean = word.strip("'\"")
        if URL_RE.match(clean):
            continue
        if clean.startswith(("/", "./", "../", "~/")) or "/" in clean or PATH_EXT_RE.search(clean):
            result.append(clean)
    return unique(result)


def detect_operators(command: str) -> list[str]:
    _, ops = scan_tokens(command)
    return unique(ops)


def detect_redirections(command: str) -> list[str]:
    redirs: list[str] = []
    quote: str | None = None
    i = 0
    while i < len(command):
        ch = command[i]
        if quote:
            if ch == "\\" and quote == '"':
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "'\"":
            quote = ch
            i += 1
            continue
        m = re.match(r"(?:\d+)?(?:&>>|&>|>>|<<|<<<|>|<)", command[i:])
        if m:
            redirs.append(m.group(0))
            i += len(m.group(0))
        else:
            i += 1
    return unique(redirs)


def detect_glob(command: str) -> bool:
    escaped = re.sub(r"\\[*?\[]", "", command)
    return any(c in escaped for c in "*?[")


def shell_features(command: str, operators: list[str]) -> dict[str, Any]:
    return {
        "has_command_substitution": "$ (".replace(" ", "") in command or "`" in command,
        "has_arithmetic_expansion": "$((" in command,
        "has_parameter_expansion": "${" in command,
        "has_glob": detect_glob(command),
        "has_quotes": "'" in command or '"' in command,
        "has_subshell": bool(re.search(r"(?:^|\s)\([^)]*\)(?:\s|$)", command)),
        "has_condition_test": "[[" in command or bool(re.search(r"(?:^|\s)\[(?:\s|$)", command)),
        "has_function_definition": bool(re.search(r"(?:^|\s)[A-Za-z_][A-Za-z0-9_]*\s*\(\)\s*\{", command)),
        "has_loop": bool(re.search(r"\b(?:for|while|until)\b", command)),
        "has_case": bool(re.search(r"\bcase\b.*\besac\b", command)),
        "has_process_substitution": "<(" in command or ">("
        in command,
        "has_here_string": "<<<" in command,
        "has_background": "&" in operators and "&&" not in operators and "||" not in operators,
    }


def classify_flags(program: str | None, flags: list[str]) -> dict[str, bool]:
    spec = PROGRAM_FLAG_RISK_MAP.get(program or "", {})
    recursive = any(f in spec.get("recursive", set()) for f in flags)
    force = any(f in spec.get("force", set()) for f in flags)
    privileged = any(f in FLAG_GROUPS["privileged"] for f in flags)
    return {"recursive": recursive, "force": force, "privileged": privileged}


def find_subcommand(program: str | None, words: list[str]) -> tuple[str | None, str]:
    if not program or program not in SUBCOMMANDS:
        return None, "none"
    allowed = SUBCOMMANDS[program]
    if program == "docker" and "compose" in words:
        ci = words.index("compose")
        for token in words[ci + 1:]:
            if token.startswith("-"):
                continue
            return f"compose {token}", "nested_table"
    try:
        idx = words.index(program)
    except ValueError:
        return None, "none"
    skip_next = False
    for token in words[idx + 1:]:
        if skip_next:
            skip_next = False
            continue
        if token == "--":
            continue
        if token.startswith("-"):
            if token in {"-C", "-c", "-m", "-n", "-p", "-u", "-f", "--format", "--message", "--name", "--output"}:
                skip_next = True
            continue
        if token in allowed:
            return token, "table"
        return None, "unknown_positional"
    return None, "none"


def infer_subcommand_operation(program: str, subcommand: str, base: tuple[str, str, str]) -> tuple[str, str, str]:
    domain, operation, detail = base
    if program == "git":
        if subcommand in {"log","show","diff","status","describe","ls-files","grep","rev-parse","shortlog","branch","tag","remote"}:
            return domain, "inspect", f"git_{subcommand}"
        if subcommand in {"reset","clean","rm"}:
            return domain, "delete_or_modify", f"git_{subcommand}"
        if subcommand in {"add","commit","checkout","switch","merge","rebase","stash","cherry-pick","mv","config","worktree","revert","restore"}:
            return domain, "modify_or_execute", f"git_{subcommand}"
        if subcommand in {"clone","fetch","pull","push"}:
            return domain, "version_control", f"git_{subcommand}"
        return domain, "version_control", f"git_{subcommand}"
    if program in {"docker","kubectl","helm"}:
        if subcommand in {"ps","images","inspect","logs","top","stats","get","describe","list","show","status","history","version","info"}:
            return domain, "inspect", f"{program}_{subcommand}"
        if subcommand in {"rm","rmi","delete","uninstall","destroy","kill"}:
            return domain, "delete_or_modify", f"{program}_{subcommand}"
        if subcommand in {"run","exec","apply","create","edit","patch","scale","set","restart","start","stop","update","upgrade","install","publish","push","pull"}:
            return domain, "modify_or_execute", f"{program}_{subcommand}"
    destructive = {"destroy","delete","remove","rm","rmi","reset","purge","rollback","kill"}
    write = {"add","commit","push","pull","clone","fetch","merge","rebase","apply","create","update","edit","install","upgrade","run","exec","start","stop","restart","delete","remove","rm","rmi","publish","deploy","import","restore","rollback","destroy"}
    read = {"status","show","inspect","get","list","ls","log","diff","describe","history","info","version","config","remote","branch","tag","output","providers","validate","fmt","plan","search"}
    if subcommand in destructive: return domain, "delete_or_modify", f"{program}_{subcommand}"
    if subcommand in write: return domain, "modify_or_execute", f"{program}_{subcommand}"
    if subcommand in read: return domain, "inspect", f"{program}_{subcommand}"
    return domain, operation, detail


def infer_operation(program: str | None, subcommand: str | None, words: list[str], command: str) -> tuple[str, str, str]:
    if not program:
        if any(ASSIGN_RE.match(w) for w in words):
            return "shell", "set_variable", "shell_variable_assignment"
        if any(x in command for x in ("[[", "]]", "case ", " esac", "if ", " then ", "for ", "while ", "until ", " function ", "() {")):
            return "shell", "control_structure", "shell_control_structure"
        if command.lstrip().startswith(("(", "{")):
            return "shell", "control_structure", "subshell_or_group"
        return "shell", "unknown", "no_executable_program"
    base = DIRECT_OPERATIONS.get(program, ("other", "unknown", program))
    domain, operation, detail = base
    if program.startswith(("./", "../", "/")) or program in {"long_running_command", "command1", "backup", "greet"}:
        return "development", "execute_program", "executable_or_script"

    if subcommand:
        domain, operation, detail = infer_subcommand_operation(program, subcommand, base)
        if program in {"aws", "gcloud", "az"}:
            operation, detail = "cloud_operation", f"{program}_{subcommand}"
        elif program == "terraform":
            operation, detail = "infrastructure_operation", f"terraform_{subcommand}"
        elif program in {"gradle", "dotnet"}:
            operation, detail = "build_or_test", f"{program}_{subcommand}"
        elif program == "redis-cli":
            operation, detail = "database_operation", f"redis_{subcommand.lower()}"
        elif program in {"systemctl", "service"} and subcommand in {"start","stop","restart","reload","try-restart","reload-or-restart","enable","disable","mask","unmask","daemon-reload","reset-failed","kill","reboot","poweroff","halt","suspend","hibernate"}:
            operation, detail = "manage_service", f"{program}_{subcommand}"
        elif program in {"apt","apt-get","dpkg"} and subcommand in {"install","remove","purge","autoremove","update","upgrade","full-upgrade","dist-upgrade","source","build-dep","download"}:
            operation, detail = "package_management", f"{program}_{subcommand}"

    if program in {"grep", "egrep", "fgrep"}:
        if "-v" in words or "--invert-match" in words:
            operation, detail = "filter_text", "inverse_text_filter"
    elif program == "find":
        if "-delete" in words:
            operation, detail = "delete", "filesystem_search_delete"
        elif "-exec" in words:
            operation, detail = "search_and_execute", "filesystem_search_exec"
        elif any(x in words for x in ("-perm", "-user", "-group")):
            operation, detail = "search", "attribute_search"
    elif program == "sed":
        if any(w.startswith(("s/", "s'", 's"')) for w in words):
            operation, detail = "transform_text", "substitution"
        if "-i" in words or "--in-place" in words:
            operation, detail = "modify_text", "in_place_edit"
    elif program == "curl":
        for i, w in enumerate(words):
            if w in {"-X", "--request"} and i + 1 < len(words):
                method = words[i + 1].upper()
                if method in {"POST", "PUT", "PATCH", "DELETE"}:
                    operation, detail = "network_request", f"http_{method.lower()}"
                break
        if operation == "network_request" and any(w in words for w in {"-o", "--output"}):
            detail = "http_download"
    elif program == "command":
        if "-v" in words or "-V" in words or "-V" in words:
            operation, detail = "inspect_command", "command_lookup"
        elif "-V" in words:
            operation, detail = "inspect_command", "command_lookup"
    elif program == "printenv":
        operation, detail = "inspect_environment", "print_environment"
    elif program == "date":
        operation, detail = "inspect_system", "date"
    elif program == "nc":
        operation, detail = "network_connection", "netcat"
    elif program in {"sha256sum","sha512sum","sha1sum","md5sum","cksum"}:
        operation = "checksum"
    elif program in {"gcc","g++","clang","clang++"}:
        operation = "compile"
    elif program in {"cargo","go","mvn"} and not subcommand:
        operation = "build"
    elif program == "jq":
        operation = "transform_json"
    elif program == "ufw" or program == "firewall-cmd" or program == "iptables":
        operation = "firewall_management"

    return domain, operation, detail


def infer_targets(program: str | None, domain: str, operation: str) -> list[str]:
    targets: list[str] = []
    if domain in {"filesystem", "disk_storage"}: targets.append("path")
    if domain == "network": targets.append("network_resource")
    if domain == "version_control": targets.append("repository_or_revision")
    if domain in {"container", "kubernetes"}: targets.append("container_or_cluster")
    if domain == "service": targets.append("service")
    if domain == "package": targets.append("package")
    if domain == "database": targets.append("database")
    if program in {"kill","pkill","killall","pgrep","pidof","ps","top","htop"}: targets.append("process")
    if operation in {"set_environment","unset_environment","inspect_environment"}: targets.append("environment")
    return unique(targets)


def infer_argument_roles(program: str | None, words: list[str]) -> list[dict[str, str]]:
    if not program: return []
    roles: list[dict[str, str]] = []
    seen_program = False
    for word in words:
        if not seen_program:
            if word == program: seen_program = True
            continue
        if word.startswith("-"):
            role = "flag"
        elif URL_RE.match(word):
            role = "url"
        elif ASSIGN_RE.match(word):
            role = "assignment"
        elif word.startswith(("/", "./", "../", "~/")) or "/" in word:
            role = "path_or_target"
        elif program in {"grep","sed","awk","cut","tr","jq"}:
            role = "expression_or_pattern"
        else:
            role = "argument"
        roles.append({"value": word, "role": role})
    return roles


def infer_risk(program: str | None, subcommand: str | None, words: list[str], command: str, flags: dict[str, bool], domain: str, operation: str, shell: dict[str, Any]) -> dict[str, Any]:
    """Infer program-aware state/risk signals.  Flags are never interpreted globally."""
    privileged = flags.get("privileged", False) or bool(re.match(r"^\s*(?:sudo|doas)(?:\s|$)", command))
    # Docker -f is contextual: logs -f means follow, not force. Only
    # deletion/update subcommands use -f as a destructive force switch.
    contextual_force = flags.get("force", False)
    if program == "docker" and subcommand in {"logs", "compose logs"}:
        contextual_force = False
    destructive = False
    modifies_data = False
    modifies_permissions = False
    execution_side_effect = False

    # Destructive filesystem/process operations.
    if program in {"rm", "rmdir", "shred", "wipefs", "userdel", "groupdel", "kill", "pkill", "killall"}:
        destructive = True
    elif program in {"mkfs", "mkfs.ext4", "mkfs.xfs", "mkfs.vfat"}:
        destructive = True
    elif program == "dd":
        destructive = any(w.startswith("of=/dev/") for w in words)
    elif program in {"fdisk", "parted", "gdisk", "cfdisk"}:
        destructive = not any(w in {"-l", "--list", "print", "p"} for w in words)
    elif program == "find":
        destructive = "-delete" in words
    elif program == "sed":
        destructive = "-i" in words or "--in-place" in words

    # Git: dry-runs are read-only; history/worktree mutations are state-changing.
    elif program == "git":
        if subcommand == "clean":
            # Git accepts compact short options such as -nd / -dn. Any -n
            # in the clean option cluster means dry-run and therefore no
            # destructive filesystem action is performed.
            clean_dry_run = any(
                w in {"--dry-run", "-n"} or
                (w.startswith("-") and not w.startswith("--") and "n" in w[1:])
                for w in words
            )
            destructive = not clean_dry_run
        elif subcommand == "reset":
            destructive = "--hard" in words
            modifies_data = True
        elif subcommand == "rm":
            destructive = True
        elif subcommand == "branch":
            if any(w in {"-D", "--delete", "--force"} for w in words):
                destructive = "-D" in words or "--force" in words
            if any(w in {"-d", "--delete", "-D", "--force"} for w in words):
                modifies_data = True
        elif subcommand == "remote" and any(w in {"add", "remove", "rename", "set-url"} for w in words):
            modifies_data = True
        elif subcommand == "config":
            # --get/--list/--show-* are read-only; setters/unsetters mutate config.
            read_only = any(w in {"--get", "--get-all", "--get-regexp", "--list", "-l", "--show-origin", "--show-scope", "--name-only"} or w.startswith("--get-") for w in words)
            if not read_only:
                modifies_data = True
        elif subcommand == "stash":
            action = next((w for w in words if w in {"push", "pop", "apply", "drop", "clear", "create", "store", "branch", "list", "show"}), None)
            if action not in {"list", "show"}:
                modifies_data = True
        elif subcommand == "worktree":
            action = next((w for w in words if w in {"list", "add", "remove", "move", "prune", "lock", "unlock"}), None)
            if action != "list":
                modifies_data = True
        elif subcommand in {"add", "commit", "checkout", "switch", "merge", "rebase", "cherry-pick", "mv", "revert", "restore"}:
            modifies_data = True
        elif subcommand in {"clone", "fetch", "pull", "push"}:
            modifies_data = True
            if subcommand == "push" and any(w in {"--force", "--force-with-lease"} for w in words):
                destructive = True
                execution_side_effect = True

    # Containers / Kubernetes.  Management actions are state-changing even when
    # not destructive; delete/kill/prune are destructive.
    elif program in {"docker", "kubectl", "helm"}:
        if subcommand in {"rm", "rmi", "delete", "destroy", "uninstall", "kill", "prune"}:
            destructive = True
        elif subcommand in {"run", "exec", "build", "create", "edit", "patch", "scale", "set", "restart", "start", "stop", "update", "upgrade", "install", "publish", "push", "pull", "apply", "connect", "disconnect", "network", "volume"}:
            modifies_data = True
            execution_side_effect = subcommand in {"run", "exec", "apply", "edit", "patch", "create"}
        if program == "kubectl" and subcommand == "rollout" and any(w == "restart" for w in words):
            modifies_data = True
            execution_side_effect = True

    elif program == "terraform":
        if subcommand == "destroy":
            destructive = True
            execution_side_effect = True
        elif subcommand in {"apply", "import", "taint", "untaint"}:
            modifies_data = True
            execution_side_effect = True

    # Package managers: installs/upgrades/updates mutate system/package state.
    elif program in {"apt", "apt-get", "dpkg", "npm", "pip", "pip3", "snap", "flatpak", "gem", "cargo"}:
        if subcommand in {"remove", "purge", "autoremove", "uninstall"}:
            destructive = True
        elif subcommand in {"install", "upgrade", "update", "full-upgrade", "dist-upgrade", "build-dep", "download", "ci", "update", "link", "publish"}:
            modifies_data = True

    # Service/system configuration.
    if program in {"systemctl", "service"} and subcommand in {"start", "stop", "restart", "reload", "enable", "disable", "mask", "unmask", "daemon-reload", "reset-failed", "kill", "reboot", "poweroff", "halt", "suspend", "hibernate"}:
        modifies_data = True
        execution_side_effect = True

    # Identity/account state. chage -l is inspection; other account-aging flags mutate.
    if program == "chage":
        if "-l" not in words and "--list" not in words:
            modifies_data = True
            modifies_permissions = True

    # File installation/copy/move/creation.
    if program in {"cp", "mv", "touch", "mkdir", "ln", "tee", "truncate", "install", "tar", "zip", "unzip", "dd"}:
        modifies_data = True
    if program in {"chmod", "chown", "chgrp", "setfacl", "setcap", "usermod", "useradd", "userdel", "groupadd", "groupdel", "passwd"}:
        modifies_data = True
        modifies_permissions = True

    if program in {"mount", "umount"}:
        modifies_data = True
        execution_side_effect = True

    if program in {"ufw", "firewall-cmd", "iptables", "ip6tables", "nft"}:
        # Status/list commands are read-only; rule changes are state-changing.
        if any(x in words for x in {"allow", "deny", "add", "delete", "remove", "flush", "enable", "disable", "--add-service", "--remove-service", "--add-port", "--remove-port"}):
            modifies_data = True
            if any(x in words for x in {"delete", "remove", "flush", "disable", "--remove-service", "--remove-port"}):
                destructive = True

    # Commands whose executable/subcommand structure is hidden behind wrappers.
    # These are genuine state changes even when the top-level program is python/aws/gcloud.
    if re.search(r"\b(?:python|python3)\s+-m\s+pip\s+(?:install|uninstall)\b", command) and not re.search(r"--dry-run\b", command):
        modifies_data = True
        if "uninstall" in command: destructive = True
    if program == "docker" and re.search(r"\bdocker\s+(?:container\s+)?kill\b", command):
        modifies_data = True
    if program == "docker" and re.search(r"\bdocker\s+(?:container\s+)?cp\b", command):
        modifies_data = True
    if program == "kubectl" and re.search(r"\bkubectl\s+cp\b", command):
        modifies_data = True
    if program in {"aws", "gcloud", "az"} and re.search(r"\b(?:aws\s+s3|gcloud\s+storage)\s+cp\b", command):
        modifies_data = True
    if program == "docker" and re.search(r"\bdocker\s+compose\s+run\b", command):
        modifies_data = True
    if program and program.startswith("mkfs"):
        destructive = True
        execution_side_effect = True

    # Nested Docker Compose actions are encoded as "docker compose <action>"
    # rather than a top-level subcommand. Treat the nested action explicitly.
    if program == "docker" and re.search(r"\bdocker\s+compose\s+(?:up|down|restart|build|run|pull|push|rm|stop|start|create)\b", command):
        modifies_data = True
        execution_side_effect = True
        if re.search(r"\bdocker\s+compose\s+(?:down|rm)\b", command):
            destructive = True

    # Cloud CLIs have nested service/action verbs that must not be left as
    # read-only merely because the top-level program is aws/gcloud/az.
    if re.search(r"\baws\s+s3\s+sync\b", command):
        modifies_data = True
        execution_side_effect = True
    if re.search(r"\baws\s+ec2\s+(?:start-instances|stop-instances|reboot-instances)\b", command):
        modifies_data = True
        execution_side_effect = True
    if re.search(r"\baws\s+ec2\s+terminate-instances\b", command):
        modifies_data = True
        destructive = True
        execution_side_effect = True
    if re.search(r"\bgcloud\s+compute\s+instances\s+(?:start|stop|reset)\b", command):
        modifies_data = True
        execution_side_effect = True
    if re.search(r"\bgcloud\s+compute\s+instances\s+(?:delete)\b", command):
        modifies_data = True
        destructive = True
        execution_side_effect = True
    if re.search(r"\baz\s+vm\s+(?:start|stop|restart)\b", command):
        modifies_data = True
        execution_side_effect = True
    if re.search(r"\baz\s+vm\s+(?:delete)\b", command):
        modifies_data = True
        destructive = True
        execution_side_effect = True

    # Firewall removal flags are program-specific and often use --remove-X=value.
    if program in {"ufw", "firewall-cmd", "iptables", "ip6tables", "nft"} and any(
        w.startswith(("--remove", "--delete")) or w in {"remove", "delete", "flush"} for w in words
    ):
        modifies_data = True
        destructive = True

    network_operation = (
        domain == "network"
        or (program == "git" and subcommand in {"clone", "fetch", "pull", "push"})
        or (program == "kubectl" and subcommand == "port-forward")
        or (program in {"ssh", "scp", "rsync", "curl", "wget", "nc", "socat"})
    )
    security_sensitive = modifies_permissions or program in {"ssh", "gpg", "openssl", "ssh-keygen", "setcap", "getcap", "ufw", "iptables", "firewall-cmd", "nft"}
    actions=[]
    if privileged: actions.append("privilege_use")
    if modifies_permissions: actions.append("permission_or_identity_change")
    if program in {"ssh", "scp", "rsync"}: actions.append("remote_access_or_transfer")
    if program in {"curl", "wget"} and any(x in w.lower() for w in words for x in ("password", "token", "authorization")): actions.append("credential_or_secret_transport")
    if program in {"openssl", "gpg", "ssh-keygen"}: actions.append("cryptographic_operation")
    if destructive: actions.append("destructive_action")
    if execution_side_effect: actions.append("state_changing_execution")
    return {"recursive": flags.get("recursive", False), "force": flags.get("force", False), "privileged": privileged, "destructive": destructive, "modifies_data": modifies_data, "modifies_permissions": modifies_permissions, "security_sensitive": security_sensitive, "security_actions": unique(actions), "network_operation": network_operation, "external_execution": bool(program and program not in SHELL_BUILTINS), "has_shell_control": shell["has_loop"] or shell["has_case"] or shell["has_condition_test"] or shell["has_function_definition"]}



def split_command_segments(command: str) -> list[str]:
    segments=[]; buf=[]; quote=None; depth=0; i=0; separators=("2>&1","2>&-","&>>","&>","&&","||","|&",";","|","&")
    while i < len(command):
        ch=command[i]
        if quote:
            buf.append(ch)
            if ch=="\\" and quote=='"' and i+1<len(command): i+=1; buf.append(command[i])
            elif ch==quote: quote=None
            i+=1; continue
        if ch in "\'\"": quote=ch; buf.append(ch); i+=1; continue
        if ch=="(": depth+=1; buf.append(ch); i+=1; continue
        if ch==")": depth=max(0,depth-1); buf.append(ch); i+=1; continue
        matched=None
        if depth==0:
            for sep in separators:
                if command.startswith(sep,i): matched=sep; break
        if matched:
            text="".join(buf).strip()
            if text: segments.append(text)
            buf.clear(); i+=len(matched); continue
        buf.append(ch); i+=1
    text="".join(buf).strip()
    if text: segments.append(text)
    return segments

def first_command_word(segment: str) -> str | None:
    text=segment.strip()
    while text.startswith("(") or text.startswith("{"):
        text=text[1:].lstrip()
    if re.match(r"^(?:for|while|until|case|function|if)\b", text):
        return None
    words,_=scan_tokens(text); i=0
    value_opts={"-u","-g","-h","-p","-n","-c","-w","-E","-o","-i","-e","-k","-s","-P","-L","-I","-d","-a"}
    while i<len(words):
        w=words[i]
        if ASSIGN_RE.match(w): i+=1; continue
        if w in WRAPPERS:
            wrapper=w; i+=1
            while i<len(words):
                opt=words[i]
                if opt in WRAPPER_VALUE_OPTIONS.get(wrapper,set()) or opt in value_opts:
                    i+=2; continue
                if wrapper=="timeout" and re.fullmatch(r"\d+(?:\.\d+)?[smhd]",opt):
                    i+=1; continue
                if opt.startswith("--signal=") or opt.startswith("--kill-after="):
                    i+=1; continue
                if opt.startswith("-"):
                    i+=1; continue
                break
            if i>=len(words): return wrapper
            continue
        if w in {"if","then","else","elif","fi","for","while","until","do","done","case","esac","in","function","{"}: i+=1; continue
        if w.startswith("("):
            w=w.lstrip("(")
        w=w.rstrip(")}")
        if w: return w
        i += 1
    return None


def extract_embedded_commands(command: str) -> list[str]:
    """Return executable command fragments used by find -exec / xargs."""
    found=[]
    m=re.search(r"(?:^|\s)-exec(?:dir)?\s+([^;]+?)(?:\s+\+|\s*;|$)",command)
    if m: found.append(m.group(1).strip())
    # SSH/SCP-style remote command: ssh host 'command'. The remote command
    # is semantically part of the user's request and must be audited too.
    mssh = re.search(r"\bssh\b(?:\s+[^'\"]+)?\s+['\"](.+?)['\"]\s*$", command)
    if mssh:
        found.append(mssh.group(1).strip())
    for xm in re.finditer(r"(?:^|[|;&]\s*)xargs\b([^|;&]*)",command):
        words,_=scan_tokens("xargs "+xm.group(1)); i=1
        value_opts={"-n","-s","-P","-L","-I","-d","-a","--max-args","--max-procs","--replace","--delimiter","--arg-file"}
        cmd=[]
        while i<len(words):
            w=words[i]
            if w in value_opts: i+=2; continue
            if w.startswith("-"): i+=1; continue
            cmd=words[i:]; break
        if cmd: found.append(" ".join(cmd))
    return unique(found)

def extract_embedded_programs(command: str) -> list[str]:
    return unique([p for frag in extract_embedded_commands(command) if (p:=first_command_word(frag))])

def extract_command_substitutions(command: str) -> list[str]:
    found=[]
    for m in re.finditer(r"\$\(([^()]*)\)", command):
        found.append(m.group(1))
    for m in re.finditer(r"`([^`]*)`", command):
        found.append(m.group(1))
    return found


def _normalize_control_segment(segment: str) -> str:
    s = segment.strip()
    # Peel grouping parentheses repeatedly when they wrap the complete segment.
    changed = True
    while changed and len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        changed = True
        depth = 0
        quote = None
        balanced = True
        for i, ch in enumerate(s):
            if quote:
                if ch == quote:
                    quote = None
                continue
            if ch in "\"'":
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    balanced = False
                    break
        if balanced and depth == 0:
            s = s[1:-1].strip()
        else:
            changed = False
    # Shell control keywords are syntax, not executable programs. Remove a
    # leading/trailing keyword so the real command becomes a stage.
    s = re.sub(r"^\s*(?:if|then|else|elif|do)\b", "", s).strip()
    s = re.sub(r"\b(?:fi|done|esac)\s*$", "", s).strip()
    return s


def build_stage_records(command: str) -> list[dict[str, Any]]:
    stages=[]
    raw_segments = split_command_segments(command)
    segments = []
    for raw in raw_segments:
        normalized = _normalize_control_segment(raw)
        if not normalized:
            continue
        inner = split_command_segments(normalized)
        if len(inner) > 1:
            segments.extend(_normalize_control_segment(x) for x in inner if _normalize_control_segment(x))
        else:
            segments.append(normalized)
    for segment in segments:
        if not segment:
            continue
        words,_=lexical_words(segment)
        words=[w.strip("(){}").strip() for w in words if w.strip("(){}").strip()]
        program=first_command_word(segment)
        if not program or program in {"for","while","until","if","case","function","{"}: continue
        subcommand,_=find_subcommand(program,words); flags=extract_flags(words); ff=classify_flags(program,flags)
        domain,operation,detail=infer_operation(program,subcommand,words,segment)
        stages.append({"program":program,"subcommand":subcommand,"domain":domain,"operation":operation,"operation_detail":detail,"flags":flags,"risk_flags":ff,"segment":segment})
    for nested in extract_command_substitutions(command):
        for nested_stage in build_stage_records(nested):
            if nested_stage["program"] not in {x["program"] for x in stages}:
                nested_stage["embedded"] = True
                stages.append(nested_stage)
    for embedded_command in extract_embedded_commands(command):
        nested_stages = build_stage_records(embedded_command)
        for nested_stage in nested_stages:
            key = (nested_stage.get("program"), nested_stage.get("segment"))
            if not any((x.get("program"), x.get("segment")) == key for x in stages):
                nested_stage["embedded"] = True
                stages.append(nested_stage)
    return stages

def enrich_record(record: dict[str, Any]) -> dict[str, Any]:
    command=str(record.get("command","")).strip(); words,word_source=lexical_words(command); operators=detect_operators(command); redirections=detect_redirections(command)
    stages=build_stage_records(command)
    if shell_features(command, operators)["has_function_definition"]:
        stages=[]
    programs=unique(x["program"] for x in stages); program=programs[0] if programs else None
    subcommand,subcommand_source=find_subcommand(program,words); flags=extract_flags(words); assignments=extract_assignments(words); shell=shell_features(command,operators); flag_features=classify_flags(program,flags)
    domain,operation,detail=infer_operation(program,subcommand,words,command); targets=infer_targets(program,domain,operation)
    if words and ASSIGN_RE.match(words[0]):
        program=None; subcommand=None; subcommand_source="assignment"
        domain,operation,detail="shell","set_variable","shell_variable_assignment"
        targets=[]
    stage_risks=[]
    for st in stages:
        stage_command=st.get("segment",st["program"])
        stage_words,_=lexical_words(stage_command)
        stage_risks.append(infer_risk(st["program"],st.get("subcommand"),stage_words,stage_command,st["risk_flags"],st["domain"],st["operation"],shell))
    if not stage_risks: stage_risks=[infer_risk(program,subcommand,words,command,flag_features,domain,operation,shell)]
    risk={"recursive":any(r["recursive"] for r in stage_risks),"force":any(r["force"] for r in stage_risks),"privileged":any(r["privileged"] for r in stage_risks),"destructive":any(r["destructive"] for r in stage_risks),"modifies_data":any(r["modifies_data"] for r in stage_risks),"modifies_permissions":any(r["modifies_permissions"] for r in stage_risks),"security_sensitive":any(r["security_sensitive"] for r in stage_risks),"security_actions":unique(a for r in stage_risks for a in r.get("security_actions",[])),"network_operation":any(r["network_operation"] for r in stage_risks),"external_execution":any(r["external_execution"] for r in stage_risks),"has_shell_control":shell["has_loop"] or shell["has_case"] or shell["has_condition_test"] or shell["has_function_definition"],"stage_count":len(stages),"destructive_privileged_stage":any(r["destructive"] and r["privileged"] for r in stage_risks),"destructive_amplified_stage":any(r["destructive"] and (r["force"] or r["recursive"]) for r in stage_risks),"stage_risks":stage_risks}
    structure_type="pipeline" if any(x in operators for x in ("|","|&")) else "compound" if any(x in operators for x in ("&&","||",";",";;",";&",";;&")) or shell["has_loop"] or shell["has_case"] or shell["has_condition_test"] or shell["has_function_definition"] else "simple_command"
    enriched=dict(record); enriched["enrichment"]={"schema_version":SCHEMA_VERSION,"program":program,"raw_program":program,"program_type":"builtin" if program in SHELL_BUILTINS else "wrapper" if program in WRAPPERS else "external" if program else "shell_construct","wrappers":[],"wrapper_arguments":[],"subcommand":subcommand,"subcommand_source":subcommand_source,"domain":domain,"operation":operation,"operation_detail":detail,"target_types":targets,"paths":extract_paths(words),"environment_variables":extract_environment_variables(command),"assignments":assignments,"flags":flags,"argument_roles":infer_argument_roles(program,words),"command_structure":{"type":structure_type,"pipeline_length":len(stages) if structure_type=="pipeline" else 1,"programs":programs,"operators":operators,"redirections":redirections,"word_source":word_source,"program_source":"stage_lexical","stages":stages},"shell_features":shell,"risk_features":risk,"parser_status":"collector_bashlex_parsed","domain_action":detail}
    return enriched


def validate_record(record: dict[str, Any]) -> None:
    if not isinstance(record, dict): raise ValueError("record is not an object")
    if "id" not in record or "command" not in record: raise ValueError("missing id/command")
    e = record.get("enrichment")
    if not isinstance(e, dict): raise ValueError("missing enrichment")
    required = {"schema_version","program","domain","operation","command_structure","risk_features","parser_status"}
    missing = required - e.keys()
    if missing: raise ValueError(f"missing enrichment fields: {sorted(missing)}")
    if not isinstance(e["command_structure"], dict): raise ValueError("command_structure must be object")
    if not isinstance(e["risk_features"], dict): raise ValueError("risk_features must be object")
    if not isinstance(e["command_structure"].get("programs"), list): raise ValueError("programs must be list")


def process(input_path: Path = INPUT_FILE, output_path: Path = OUTPUT_FILE) -> dict[str, Any]:
    if not input_path.exists(): raise FileNotFoundError(f"Input file not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    total = success = failed = 0
    stats = {k: Counter() for k in ("program","operation","domain","structure","parser_status")}
    failures: list[tuple[int,str]] = []
    with input_path.open("r", encoding="utf-8") as src, temp.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, 1):
            if not line.strip(): continue
            total += 1
            try:
                rec = json.loads(line)
                out = enrich_record(rec)
                validate_record(out)
                dst.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
                success += 1
                e = out["enrichment"]
                stats["program"][str(e["program"])] += 1
                stats["operation"][e["operation"]] += 1
                stats["domain"][e["domain"]] += 1
                stats["structure"][e["command_structure"]["type"]] += 1
                stats["parser_status"][e["parser_status"]] += 1
            except Exception as exc:
                failed += 1; failures.append((line_no, str(exc)))
    if failed:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"Enrichment failed on {failed} records; first failures: {failures[:10]}")
    temp.replace(output_path)
    return {"total": total, "success": success, "failed": failed, **stats}


def audit_output(path: Path, expected: int) -> dict[str, Any]:
    rows = 0; malformed = 0; duplicate_ids = 0; ids: set[Any] = set(); validation_errors = Counter()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip(): continue
            try:
                r = json.loads(line); rows += 1
                if r.get("id") in ids: duplicate_ids += 1
                ids.add(r.get("id"))
                try: validate_record(r)
                except Exception as exc: validation_errors[str(exc)] += 1
            except Exception: malformed += 1
    return {"rows": rows, "expected": expected, "malformed": malformed, "duplicate_ids": duplicate_ids, "validation_errors": validation_errors}


def semantic_tests() -> None:
    cases = {
        "rm -rf /tmp/build": ("rm", "delete"),
        "sudo systemctl restart nginx": ("systemctl", "manage_service"),
        "git status": ("git", "inspect"),
        "git reset --hard HEAD~1": ("git", "delete_or_modify"),
        "env | sort": ("env", "inspect_environment"),
        "printenv HOME": ("printenv", "inspect_environment"),
        "command -v python3": ("command", "inspect_command"),
        "curl -X DELETE https://example.com/users/1": ("curl", "network_request"),
        "find . -type f -exec ls {} +": ("find", "search_and_execute"),
        "echo $((10 + 20))": ("echo", "output"),
        "files=$(ls)": (None, "set_variable"),
    }
    for command, expected in cases.items():
        words, _ = lexical_words(command)
        rec = {"id": 1, "command": command, "commands": words}
        e = enrich_record(rec)["enrichment"]
        assert e["program"] == expected[0], (command, e["program"])
        assert e["operation"] == expected[1], (command, e["operation"])


def main() -> None:
    semantic_tests()
    result = process()
    audit = audit_output(OUTPUT_FILE, result["success"])
    print("=" * 68)
    print("SafeShell Enrichment V11")
    print("=" * 68)
    print(f"Input       : {INPUT_FILE}")
    print(f"Output      : {OUTPUT_FILE}")
    print(f"Total       : {result['total']}")
    print(f"Successful  : {result['success']}")
    print(f"Failed      : {result['failed']}")
    print(f"Output rows : {audit['rows']}")
    print(f"Malformed   : {audit['malformed']}")
    print(f"Duplicate IDs: {audit['duplicate_ids']}")
    print("Structures:")
    for k,v in result["structure"].most_common(): print(f"  {k}: {v}")
    print("Top programs:")
    for k,v in result["program"].most_common(20): print(f"  {k}: {v}")
    print("Top operations:")
    for k,v in result["operation"].most_common(30): print(f"  {k}: {v}")
    print("Top unknown operations:")
    unknown = Counter()
    with OUTPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)["enrichment"]
            if e["operation"] == "unknown": unknown[str(e["program"])] += 1
    for k,v in unknown.most_common(20): print(f"  {k}: {v}")
    if audit["rows"] != result["success"] or audit["malformed"] or audit["duplicate_ids"] or audit["validation_errors"]:
        raise SystemExit("FINAL AUDIT FAILED")
    print("Validation  : PASS")
    print("Semantic tests: PASS")
    print("=" * 68)


if __name__ == "__main__":
    main()