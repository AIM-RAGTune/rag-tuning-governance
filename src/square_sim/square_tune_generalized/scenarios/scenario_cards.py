from __future__ import annotations


def scenario_card_text(track: str, scenario: str, source_datasets: list[str]) -> str:
    caveat = "Healthcare operations proxy only; not clinical diagnosis or treatment planning." if track == "patient_flow" else "Software benchmark scenario; not hardware validation."
    return "\n".join(
        [
            f"# Scenario Card: {scenario}",
            "",
            f"Track: `{track}`",
            f"Sources: {', '.join(source_datasets)}",
            "",
            "## What This Tests",
            "Adaptive policy improvement under cost, uncertainty, regression, and policy constraints.",
            "",
            "## What This Does Not Test",
            caveat,
            "It does not prove SQUARE hardware, clinical efficacy, or commercial ROI.",
            "",
            "## Baselines",
            "Static, greedy, coordinate-descent, evolutionary, optional Bayesian/Optuna, SQUARETune ablations, and adaptive compute.",
            "",
            "## Publication Caveats",
            "Restricted or credentialed raw data must not be included in publication bundles.",
        ]
    ) + "\n"
