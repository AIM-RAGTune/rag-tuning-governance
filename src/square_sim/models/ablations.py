SQUARESIM_ABLATIONS = [
    "squaresim_no_feedback",
    "squaresim_no_nonlinear",
    "squaresim_no_memory",
    "squaresim_no_overlap_zones",
    "squaresim_static_emitters",
    "squaresim_no_phase",
    "squaresim_linear_field_only",
]

SQUARESIM_SNAPSHOT_ABLATIONS = [
    "squaresim_snapshot_no_fork",
    "squaresim_snapshot_linear_rollout",
    "squaresim_snapshot_no_merge",
    "squaresim_snapshot_random_branch",
    "squaresim_snapshot_no_feedback",
    "squaresim_snapshot_whole_field",
    "squaresim_snapshot_local_only",
    "squaresim_snapshot_no_memory",
    "squaresim_snapshot_no_nonlinear",
    "squaresim_snapshot_no_phase",
]

SQUARESIM_MODELS = [
    "squaresim_full",
    *SQUARESIM_ABLATIONS,
    "squaresim_snapshot_rollout",
    *SQUARESIM_SNAPSHOT_ABLATIONS,
]
