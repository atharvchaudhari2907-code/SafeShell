import json
from pathlib import Path
from collections import Counter, defaultdict


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "enriched"
    / "enriched_commands_v2.jsonl"
)


# =========================================================
# Helpers
# =========================================================

def load_records():
    records = []

    with INPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:
                records.append(
                    json.loads(line)
                )
            except json.JSONDecodeError:
                continue

    return records


def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# =========================================================
# Main analysis
# =========================================================

def analyze():

    if not INPUT_FILE.exists():

        print(
            f"File not found: {INPUT_FILE}"
        )

        return

    records = load_records()

    print_section(
        "SafeShell Enrichment Diagnostic"
    )

    print(
        f"Total records: {len(records)}"
    )

    # -----------------------------------------------------
    # Counters
    # -----------------------------------------------------

    unknown_program_commands = Counter()

    unknown_operation_commands = Counter()

    unknown_operation_by_program = defaultdict(
        Counter
    )

    domain_counter = Counter()

    operation_counter = Counter()

    program_counter = Counter()

    target_counter = Counter()

    flag_counter = Counter()

    # -----------------------------------------------------
    # Boolean enrichment statistics
    # -----------------------------------------------------

    boolean_fields = [
        "recursive",
        "force",
        "privileged",
        "destructive",
        "modifies_data",
        "modifies_permissions",
        "modifies_system_state",
        "security_sensitive",
        "network_operation",
        "external_execution",
    ]

    boolean_counts = Counter()

    # -----------------------------------------------------
    # Process records
    # -----------------------------------------------------

    for record in records:

        enrichment = record.get(
            "enrichment",
            {}
        )

        command = record.get(
            "command",
            ""
        )

        program = enrichment.get(
            "program"
        )

        operation = enrichment.get(
            "operation",
            "unknown"
        )

        domain = enrichment.get(
            "domain",
            "unknown"
        )

        # ---------------------------------------------
        # General counters
        # ---------------------------------------------

        program_counter[
            str(program)
        ] += 1

        operation_counter[
            operation
        ] += 1

        domain_counter[
            domain
        ] += 1

        # ---------------------------------------------
        # Unknown programs
        # ---------------------------------------------

        if program is None:

            unknown_program_commands[
                command
            ] += 1

        # ---------------------------------------------
        # Unknown operations
        # ---------------------------------------------

        if operation == "unknown":

            unknown_operation_commands[
                command
            ] += 1

            unknown_operation_by_program[
                str(program)
            ][command] += 1

        # ---------------------------------------------
        # Targets
        # ---------------------------------------------

        for target in enrichment.get(
            "target_types",
            []
        ):

            target_counter[
                target
            ] += 1

        # ---------------------------------------------
        # Flags
        # ---------------------------------------------

        for flag in record.get(
            "flags",
            []
        ):

            flag_counter[
                flag
            ] += 1

        # ---------------------------------------------
        # Boolean fields
        # ---------------------------------------------

        for field in boolean_fields:

            if enrichment.get(
                field,
                False
            ):

                boolean_counts[
                    field
                ] += 1

    # =====================================================
    # 1. Unknown program records
    # =====================================================

    print_section(
        "1. Commands With Unknown Program"
    )

    print(
        f"Total records: "
        f"{sum(unknown_program_commands.values())}"
    )

    if unknown_program_commands:

        for command, count in (
            unknown_program_commands
            .most_common()
        ):

            print(
                f"{count:4}  {command}"
            )

    else:

        print("None")

    # =====================================================
    # 2. Unknown operations by program
    # =====================================================

    print_section(
        "2. Unknown Operations By Program"
    )

    unknown_program_summary = []

    for program, commands in (
        unknown_operation_by_program.items()
    ):

        unknown_program_summary.append(
            (
                program,
                sum(commands.values())
            )
        )

    unknown_program_summary.sort(
        key=lambda item: item[1],
        reverse=True
    )

    for program, count in (
        unknown_program_summary
    )[:50]:

        print(
            f"{count:4}  {program}"
        )

    # =====================================================
    # 3. Actual commands for unknown operations
    # =====================================================

    print_section(
        "3. Commands With Unknown Operation"
    )

    displayed = 0

    for program, commands in (
        sorted(
            unknown_operation_by_program.items(),
            key=lambda item: sum(
                item[1].values()
            ),
            reverse=True
        )
    ):

        print()
        print(
            f"[{program}]"
        )

        for command, count in (
            commands.most_common(20)
        ):

            print(
                f"  {count:4}  {command}"
            )

            displayed += 1

            if displayed >= 150:
                break

        if displayed >= 150:
            break

    # =====================================================
    # 4. Program distribution
    # =====================================================

    print_section(
        "4. Top Programs"
    )

    for program, count in (
        program_counter.most_common(50)
    ):

        print(
            f"{count:4}  {program}"
        )

    # =====================================================
    # 5. Domain distribution
    # =====================================================

    print_section(
        "5. Domains"
    )

    for domain, count in (
        domain_counter.most_common()
    ):

        print(
            f"{count:4}  {domain}"
        )

    # =====================================================
    # 6. Operation distribution
    # =====================================================

    print_section(
        "6. Operations"
    )

    for operation, count in (
        operation_counter.most_common(50)
    ):

        print(
            f"{count:4}  {operation}"
        )

    # =====================================================
    # 7. Target distribution
    # =====================================================

    print_section(
        "7. Target Types"
    )

    for target, count in (
        target_counter.most_common()
    ):

        print(
            f"{count:4}  {target}"
        )

    # =====================================================
    # 8. Most common flags
    # =====================================================

    print_section(
        "8. Most Common Flags"
    )

    for flag, count in (
        flag_counter.most_common(50)
    ):

        print(
            f"{count:4}  {flag}"
        )

    # =====================================================
    # 9. Boolean enrichment coverage
    # =====================================================

    print_section(
        "9. Boolean Enrichment Coverage"
    )

    total = len(records)

    for field in boolean_fields:

        count = boolean_counts[field]

        percentage = (
            count / total * 100
            if total
            else 0
        )

        print(
            f"{field:30} "
            f"{count:6} "
            f"({percentage:6.2f}%)"
        )

    # =====================================================
    # 10. Unknown operation percentage
    # =====================================================

    print_section(
        "10. Quality Summary"
    )

    unknown_program_count = sum(
        unknown_program_commands.values()
    )

    unknown_operation_count = sum(
        unknown_operation_commands.values()
    )

    print(
        f"Total records:          {total}"
    )

    print(
        f"Unknown programs:       "
        f"{unknown_program_count}"
    )

    print(
        f"Unknown operations:     "
        f"{unknown_operation_count}"
    )

    if total:

        print(
            f"Unknown program rate:   "
            f"{unknown_program_count / total * 100:.2f}%"
        )

        print(
            f"Unknown operation rate: "
            f"{unknown_operation_count / total * 100:.2f}%"
        )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    analyze()