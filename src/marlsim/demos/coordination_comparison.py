from __future__ import annotations

from marlsim.experiments import format_comparison_table, run_coordination_comparison


def main() -> None:
    results = run_coordination_comparison()
    print(format_comparison_table(results))


if __name__ == "__main__":
    main()
