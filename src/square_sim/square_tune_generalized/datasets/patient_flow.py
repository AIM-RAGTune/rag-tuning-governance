from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from square_sim.utils.hashing import stable_hash


def generate_patient_flow_synthetic_proxy(rows: int, seed: int = 101) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    acuity = rng.integers(1, 6, size=rows)
    occupancy = np.clip(rng.normal(0.78, 0.16, rows), 0, 1)
    staffing = np.clip(rng.normal(0.68, 0.13, rows), 0, 1)
    arrival = np.clip(rng.gamma(2.5, 0.16, rows), 0, 1.4)
    boarding = np.clip(
        0.25 * acuity / 5 + 0.55 * occupancy + 0.35 * arrival - 0.28 * staffing + rng.normal(0, 0.08, rows),
        0,
        1,
    )
    return pd.DataFrame(
        {
            "row_id": [f"pf-{seed}-{i}" for i in range(rows)],
            "source_dataset": "patient_flow_synthetic_proxy_v1",
            "track": "patient_flow",
            "arrival_rate": arrival,
            "acuity": acuity,
            "bed_occupancy": occupancy,
            "staffing_proxy": staffing,
            "boarding_risk": boarding,
            "admission_risk": np.clip(0.15 + acuity * 0.11 + occupancy * 0.28 + rng.normal(0, 0.06, rows), 0, 1),
            "bottleneck_label": np.where(occupancy > 0.82, "beds", np.where(staffing < 0.58, "staffing", "throughput")),
            "unsafe_recommendation_risk": np.clip(boarding * 0.25 + (1 - staffing) * 0.25, 0, 1),
            "input_text": [f"Operations state {i}: ED flow and resource context." for i in range(rows)],
        }
    )


def _first_existing(root: Path, names: list[str]) -> Path | None:
    lowered = {name.lower() for name in names}
    for path in root.rglob("*"):
        if path.is_file() and path.name.lower() in lowered:
            return path
    return None


def _read_table(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _column(frame: pd.DataFrame, *names: str) -> str | None:
    lookup = {column.lower(): column for column in frame.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def _scaled(series: pd.Series, default: float = 0.5) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series([default] * len(series), index=series.index)
    lo = float(numeric.quantile(0.05))
    hi = float(numeric.quantile(0.95))
    if hi <= lo:
        return pd.Series([default] * len(series), index=series.index)
    return ((numeric - lo) / (hi - lo)).clip(0, 1).fillna(default)


def import_mimic_patient_flow(path: Path, *, max_rows: int | None = None) -> pd.DataFrame:
    """Normalize credentialed MIMIC-IV-ED or MIMIC-IV tables into operations proxy rows.

    The output intentionally contains derived operations features only, hashed row IDs, and no raw
    subject/hospital/stay identifiers. It remains restricted because source access is credentialed.
    """
    root = path.expanduser()
    if not root.exists():
        raise FileNotFoundError(f"MIMIC path does not exist: {root}")
    edstays = _read_table(_first_existing(root, ["edstays.csv.gz", "edstays.csv", "edstays.parquet"]))
    triage = _read_table(_first_existing(root, ["triage.csv.gz", "triage.csv", "triage.parquet"]))
    admissions = _read_table(_first_existing(root, ["admissions.csv.gz", "admissions.csv", "admissions.parquet"]))
    if edstays.empty and admissions.empty:
        raise FileNotFoundError(
            "Could not find MIMIC ED `edstays` or hosp `admissions` table. "
            "Expected files such as edstays.csv.gz, triage.csv.gz, or admissions.csv.gz."
        )

    if not edstays.empty:
        frame = edstays.copy()
        stay_col = _column(frame, "stay_id")
        subject_col = _column(frame, "subject_id")
        join_cols = [col for col in [subject_col, stay_col] if col]
        if not triage.empty and join_cols:
            triage_cols = [col for col in join_cols if col in triage.columns]
            if triage_cols:
                frame = frame.merge(triage, on=triage_cols, how="left", suffixes=("", "_triage"))
        intime_col = _column(frame, "intime")
        outtime_col = _column(frame, "outtime")
        disposition_col = _column(frame, "disposition")
        acuity_col = _column(frame, "acuity")
        if intime_col and outtime_col:
            intime = pd.to_datetime(frame[intime_col], errors="coerce")
            outtime = pd.to_datetime(frame[outtime_col], errors="coerce")
            hours = ((outtime - intime).dt.total_seconds() / 3600.0).clip(lower=0)
            arrival_bucket = intime.dt.floor("h")
            arrival_rate = arrival_bucket.map(arrival_bucket.value_counts()).fillna(1)
        else:
            hours = pd.Series([4.0] * len(frame), index=frame.index)
            arrival_rate = pd.Series([1.0] * len(frame), index=frame.index)
        disposition = frame[disposition_col].astype(str).str.lower() if disposition_col else pd.Series(["unknown"] * len(frame))
        acuity = pd.to_numeric(frame[acuity_col], errors="coerce").fillna(3) if acuity_col else pd.Series([3] * len(frame))
        source_name = "mimic_iv_ed"
        raw_id = (
            frame[stay_col].astype(str)
            if stay_col
            else pd.Series([f"row-{i}" for i in range(len(frame))], index=frame.index)
        )
    else:
        frame = admissions.copy()
        subject_col = _column(frame, "subject_id")
        hadm_col = _column(frame, "hadm_id")
        intime_col = _column(frame, "admittime")
        outtime_col = _column(frame, "dischtime")
        disposition_col = _column(frame, "hospital_expire_flag", "admission_type")
        if intime_col and outtime_col:
            intime = pd.to_datetime(frame[intime_col], errors="coerce")
            outtime = pd.to_datetime(frame[outtime_col], errors="coerce")
            hours = ((outtime - intime).dt.total_seconds() / 3600.0).clip(lower=0)
            arrival_bucket = intime.dt.floor("h")
            arrival_rate = arrival_bucket.map(arrival_bucket.value_counts()).fillna(1)
        else:
            hours = pd.Series([24.0] * len(frame), index=frame.index)
            arrival_rate = pd.Series([1.0] * len(frame), index=frame.index)
        acuity = pd.Series([3] * len(frame), index=frame.index)
        disposition = frame[disposition_col].astype(str).str.lower() if disposition_col else pd.Series(["unknown"] * len(frame))
        source_name = "mimic_iv"
        raw_id = (
            frame[hadm_col].astype(str)
            if hadm_col
            else pd.Series([f"{subject_col or 'subject'}-{i}" for i in range(len(frame))], index=frame.index)
        )

    if max_rows is not None:
        frame = frame.head(max_rows)
        hours = hours.head(max_rows)
        arrival_rate = arrival_rate.head(max_rows)
        acuity = acuity.head(max_rows)
        disposition = disposition.head(max_rows)
        raw_id = raw_id.head(max_rows)

    arrival_scaled = _scaled(arrival_rate)
    boarding_risk = _scaled(hours)
    acuity_scaled = (pd.to_numeric(acuity, errors="coerce").fillna(3).clip(1, 5) / 5.0).astype(float)
    admission_risk = (
        disposition.str.contains("admit|hospital|observation|transfer", case=False, regex=True).astype(float) * 0.7
        + acuity_scaled * 0.3
    ).clip(0, 1)
    occupancy_proxy = (0.45 + 0.4 * arrival_scaled + 0.2 * boarding_risk).clip(0, 1)
    staffing_proxy = (0.85 - 0.35 * arrival_scaled).clip(0, 1)
    bottleneck = np.where(boarding_risk > 0.72, "boarding", np.where(arrival_scaled > 0.68, "arrival_surge", "throughput"))
    rows = len(raw_id)
    return pd.DataFrame(
        {
            "row_id": [f"mimic-{stable_hash({'id': value}, 16)}" for value in raw_id.astype(str)],
            "source_dataset": source_name,
            "track": "patient_flow",
            "arrival_rate": arrival_scaled.to_numpy(),
            "acuity": pd.to_numeric(acuity, errors="coerce").fillna(3).clip(1, 5).to_numpy(),
            "bed_occupancy": occupancy_proxy.to_numpy(),
            "staffing_proxy": staffing_proxy.to_numpy(),
            "boarding_risk": boarding_risk.to_numpy(),
            "admission_risk": admission_risk.to_numpy(),
            "bottleneck_label": bottleneck,
            "unsafe_recommendation_risk": np.clip(0.08 + 0.18 * boarding_risk.to_numpy() + 0.12 * acuity_scaled.to_numpy(), 0, 1),
            "input_text": [f"Credentialed operations case {i}: patient-flow context summarized from de-identified MIMIC tables." for i in range(rows)],
        }
    )
