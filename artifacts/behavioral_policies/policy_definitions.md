# Behaviorally Distinct Policy Definitions

- `low_retrieval_single_endpoint`: fewer endpoints and lower observed budget than expanded retrieval
- `expanded_retrieval_multi_endpoint`: more endpoints, more API calls, and higher observed budget
- `adaptive_routing_on_insufficient_evidence`: API calls vary by query rather than staying fixed
- `measured_cost_minimizer_at_quality_floor`: selector uses measured budget_units, not labels
- `measured_latency_minimizer_at_quality_floor`: selector uses observed latency_ms, not labels
- `quality_only_best_on_validation`: ignores measured cost and latency
- `constrained_quality_optimizer`: reports active constraints and feasible winner
- `pareto_frontier_selector`: does not collapse objectives into scalar utility
- `governed_selection`: promotion decision changes because operating constraints are enforced
