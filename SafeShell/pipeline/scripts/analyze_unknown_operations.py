import json
from pathlib import Path
from collections import Counter, defaultdict


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "enriched"
    / "enriched_commands_v3.jsonl"
)


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
                pass

    return records


def main():

    if not INPUT_FILE.exists():

        print(
            f"File not found: {INPUT_FILE}"
        )

        return

    records = load_records()

    print()
    print("=" * 70)
    print("SafeShell Unknown Operation Audit")
    print("=" * 70)

    print(
        f"Total records: {len(records)}"
    )

    # -----------------------------------------------------
    # Group unknown operations
    # -----------------------------------------------------

    by_program = defaultdict(Counter)

    unknown_commands = Counter()

    for record in records:

        enrichment = record.get(
            "enrichment",
            {}
        )

        operation = enrichment.get(
            "operation"
        )

        if operation != "unknown":
            continue

        program = enrichment.get(
            "program"
        )

        command = record.get(
            "command",
            ""
        )

        by_program[
            str(program)
        ][command] += 1

        unknown_commands[
            command
        ] += 1

    total_unknown = sum(
        unknown_commands.values()
    )

    print(
        f"Unknown operations: {total_unknown}"
    )

    # -----------------------------------------------------
    # Program summary
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("Unknown Operations By Program")
    print("=" * 70)

    program_summary = []

    for program, commands in by_program.items():

        count = sum(
            commands.values()
        )

        program_summary.append(
            (
                program,
                count
            )
        )

    program_summary.sort(
        key=lambda x: x[1],
        reverse=True
    )

    for program, count in program_summary:

        print(
            f"{count:5}  {program}"
        )

    # -----------------------------------------------------
    # Actual commands
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("Actual Unknown Commands")
    print("=" * 70)

    displayed = 0

    for program, count in program_summary:

        print()
        print(
            f"[{program}] - {count} commands"
        )

        commands = by_program[
            program
        ]

        for command, frequency in (
            commands.most_common()
        ):

            print(
                f"  {frequency:4}  {command}"
            )

            displayed += 1

            if displayed >= 500:

                print()
                print(
                    "Output limited to "
                    "500 commands."
                )

                break

        if displayed >= 500:
            break

    # -----------------------------------------------------
    # Program-level operation candidates
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("Suggested V4 Investigation Groups")
    print("=" * 70)

    important_programs = [
        "python3",
        "python",
        "node",
        "aws",
        "gcloud",
        "az",
        "psql",
        "mysql",
        "sqlite3",
        "gpg",
        "ssh",
        "ssh-keygen",
        "getcap",
        "getfacl",
        "seq",
        "dmesg",
        "ufw",
        "firewall-cmd",
        "usermod",
    ]

    for program in important_programs:

        if program not in by_program:
            continue

        commands = by_program[
            program
        ]

        print()
        print(
            f"[{program}]"
        )

        for command, frequency in (
            commands.most_common()
        ):

            print(
                f"  {frequency:4}  {command}"
            )

    # -----------------------------------------------------
    # Construct distribution of unknown operations
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("Unknown Operations By Shell Construct")
    print("=" * 70)

    construct_counter = Counter()

    for record in records:

        enrichment = record.get(
            "enrichment",
            {}
        )

        if enrichment.get(
            "operation"
        ) != "unknown":

            continue

        construct = (
            record
            .get("execution", {})
            .get(
                "construct",
                "unknown"
            )
        )

        construct_counter[
            construct
        ] += 1

    for construct, count in (
        construct_counter.most_common()
    ):

        print(
            f"{count:5}  {construct}"
        )

    # -----------------------------------------------------
    # Save machine-readable report
    # -----------------------------------------------------

    report = {
        "total_records": len(records),
        "unknown_operations": total_unknown,
        "programs": []
    }

    for program, count in program_summary:

        commands = by_program[
            program
        ]

        report["programs"].append({
            "program": program,
            "count": count,
            "commands": [
                {
                    "command": command,
                    "count": frequency
                }
                for command, frequency
                in commands.most_common()
            ]
        })

    REPORT_FILE = (
        BASE_DIR
        / "data"
        / "enriched"
        / "unknown_operations_v3.json"
    )

    with REPORT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 70)
    print("Report written to:")
    print(REPORT_FILE)
    print("=" * 70)


if __name__ == "__main__":
    main()