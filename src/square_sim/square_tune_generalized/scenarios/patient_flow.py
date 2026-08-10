from __future__ import annotations

PATIENT_FLOW_POLICY_SPACE = {
    "retrieval_context_window": [4, 8, 12],
    "operational_explanation_depth": ["brief", "standard", "detailed"],
    "escalation_threshold": [0.45, 0.60, 0.75],
    "bottleneck_alert_threshold": [0.55, 0.70],
    "bed_demand_scenario_count": [3, 5, 8],
    "staffing_sensitivity_weight": [0.25, 0.5, 0.75],
    "admission_risk_threshold": [0.4, 0.6],
    "boarding_risk_threshold": [0.45, 0.65],
    "explanation_abstention_threshold": [0.35, 0.55],
}
