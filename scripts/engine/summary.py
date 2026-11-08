from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import numpy as np
import pandas as pd

def build_sample_score_table(
    scope_df: pd.DataFrame,
    cfg: Dict[str, object],
) -> pd.DataFrame:
    sample_unit_key = str(cfg["analysis_sample_unit_key"])
    donor_key = str(cfg["donor_key"])
    analysis_key = str(cfg["analysis_axis_key"])
    raw_key = str(cfg["raw_distance_key"])
    shift_key = str(cfg["atlas_shift_key"])
    robust_key = str(cfg["atlas_shift_robust_key"])
    high_flag_key = str(cfg["atlas_high_flag_key"])
    exceedance_key = str(cfg["atlas_exceedance_key"])
    exceedance_residual_key = str(cfg["atlas_exceedance_residual_key"])
    high_exceedance_flag_key = str(cfg["atlas_high_exceedance_flag_key"])
    donor_top_quantile = float(cfg["donor_top_quantile"])

    sample_df = (
        scope_df.groupby([sample_unit_key, analysis_key], observed=True)
        .agg(
            donor_id=(donor_key, lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]),
            donor_median_raw=(raw_key, "median"),
            donor_median_shift=(shift_key, "median"),
            donor_median_shift_robust=(robust_key, "median"),
            donor_high_shift_fraction=(high_flag_key, "mean"),
            donor_top10_shift=(shift_key, lambda x: float(np.quantile(x, donor_top_quantile))),
            donor_median_exceedance=(exceedance_key, "median"),
            donor_mean_exceedance=(exceedance_key, "mean"),
            donor_median_exceedance_residual=(exceedance_residual_key, "median"),
            donor_high_exceedance_fraction=(high_exceedance_flag_key, "mean"),
            donor_top10_exceedance=(exceedance_key, lambda x: float(np.quantile(x, donor_top_quantile))),
            n_cells=(raw_key, "size"),
        )
        .reset_index()
        .rename(columns={sample_unit_key: "sample_unit"})
    )
    return sample_df

def build_shift_driver_table(
    scope_df: pd.DataFrame,
    cfg: Dict[str, object],
    disease_value: str,
    group_key: str,
    metric_key: Optional[str] = None,
    positive_cutoff: Optional[float] = None,
    high_shift_quantile: float = 0.90,
) -> pd.DataFrame:
    analysis_key = str(cfg["analysis_axis_key"])
    control_label = str(cfg["analysis_control_label"])
    sample_unit_key = str(cfg["analysis_sample_unit_key"])
    metric_key = metric_key or str(cfg.get("atlas_exceedance_key", cfg["atlas_shift_key"]))
    if positive_cutoff is None:
        positive_cutoff = 0.5 if metric_key == str(cfg.get("atlas_exceedance_key", "atlas_exceedance_score")) else 0.0

    work_df = scope_df.loc[scope_df[analysis_key].isin([control_label, str(disease_value)])].copy()
    if work_df.empty:
        return pd.DataFrame()

    control_df = work_df.loc[work_df[analysis_key].eq(control_label)].copy()
    case_df = work_df.loc[work_df[analysis_key].eq(str(disease_value))].copy()
    if control_df.empty or case_df.empty:
        return pd.DataFrame()

    control_threshold = float(control_df[metric_key].quantile(high_shift_quantile))
    control_positive_total = float(np.clip(control_df[metric_key].to_numpy(dtype=float) - positive_cutoff, 0.0, None).sum())
    case_positive_total = float(np.clip(case_df[metric_key].to_numpy(dtype=float) - positive_cutoff, 0.0, None).sum())
    control_high_total = int(np.sum(control_df[metric_key].to_numpy(dtype=float) > control_threshold))
    case_high_total = int(np.sum(case_df[metric_key].to_numpy(dtype=float) > control_threshold))
    min_sample_cells = int(cfg.get("state_summary_min_cells_per_unit", 10))

    records: List[Dict[str, object]] = []
    for group_value, group_df in work_df.groupby(group_key, observed=True, sort=False):
        group_control_df = group_df.loc[group_df[analysis_key].eq(control_label)].copy()
        group_case_df = group_df.loc[group_df[analysis_key].eq(str(disease_value))].copy()

        control_values = group_control_df[metric_key].dropna().to_numpy(dtype=float)
        case_values = group_case_df[metric_key].dropna().to_numpy(dtype=float)
        if control_values.size == 0 and case_values.size == 0:
            continue

        control_positive_mass = float(np.clip(control_values - positive_cutoff, 0.0, None).sum())
        case_positive_mass = float(np.clip(case_values - positive_cutoff, 0.0, None).sum())
        control_high_cells = int(np.sum(control_values > control_threshold))
        case_high_cells = int(np.sum(case_values > control_threshold))

        sample_metric_df = (
            group_df.groupby([sample_unit_key, analysis_key], observed=True)[metric_key]
            .agg(sample_mean_metric="mean", sample_median_metric="median", n_group_cells="size")
            .reset_index()
        )
        sample_metric_df = sample_metric_df.loc[sample_metric_df["n_group_cells"] >= min_sample_cells].copy()
        sample_control_values = sample_metric_df.loc[
            sample_metric_df[analysis_key].eq(control_label), "sample_mean_metric"
        ].to_numpy(dtype=float)
        sample_case_values = sample_metric_df.loc[
            sample_metric_df[analysis_key].eq(str(disease_value)), "sample_mean_metric"
        ].to_numpy(dtype=float)

        case_positive_share = case_positive_mass / case_positive_total if case_positive_total > 0 else np.nan
        control_positive_share = control_positive_mass / control_positive_total if control_positive_total > 0 else np.nan
        case_high_share = case_high_cells / case_high_total if case_high_total > 0 else np.nan
        control_high_share = control_high_cells / control_high_total if control_high_total > 0 else np.nan

        records.append(
            {
                group_key: str(group_value),
                "case_group": str(disease_value),
                "metric_key": metric_key,
                "n_control_cells": int(control_values.size),
                "n_case_cells": int(case_values.size),
                "n_control_units": int(sample_control_values.size),
                "n_case_units": int(sample_case_values.size),
                "sample_unit_min_cells": min_sample_cells,
                "control_median": float(np.median(control_values)) if control_values.size else np.nan,
                "case_median": float(np.median(case_values)) if case_values.size else np.nan,
                "control_mean": float(np.mean(control_values)) if control_values.size else np.nan,
                "case_mean": float(np.mean(case_values)) if case_values.size else np.nan,
                "delta_median": float(np.median(case_values) - np.median(control_values))
                if (case_values.size and control_values.size)
                else np.nan,
                "delta_mean": float(np.mean(case_values) - np.mean(control_values))
                if (case_values.size and control_values.size)
                else np.nan,
                "sample_mean_control": float(np.mean(sample_control_values)) if sample_control_values.size else np.nan,
                "sample_mean_case": float(np.mean(sample_case_values)) if sample_case_values.size else np.nan,
                "sample_mean_delta": float(np.mean(sample_case_values) - np.mean(sample_control_values))
                if (sample_case_values.size and sample_control_values.size)
                else np.nan,
                "control_positive_shift_mass": control_positive_mass,
                "case_positive_shift_mass": case_positive_mass,
                "control_positive_shift_mass_share": control_positive_share,
                "case_positive_shift_mass_share": case_positive_share,
                "excess_positive_shift_mass_share": case_positive_share - control_positive_share
                if (pd.notna(case_positive_share) and pd.notna(control_positive_share))
                else np.nan,
                "control_high_shift_cells": control_high_cells,
                "case_high_shift_cells": case_high_cells,
                "control_high_shift_fraction": float(np.mean(control_values > control_threshold)) if control_values.size else np.nan,
                "case_high_shift_fraction": float(np.mean(case_values > control_threshold)) if case_values.size else np.nan,
                "high_shift_fraction_diff": float(np.mean(case_values > control_threshold) - np.mean(control_values > control_threshold))
                if (case_values.size and control_values.size)
                else np.nan,
                "control_high_shift_share": control_high_share,
                "case_high_shift_share": case_high_share,
                "excess_high_shift_share": case_high_share - control_high_share
                if (pd.notna(case_high_share) and pd.notna(control_high_share))
                else np.nan,
            }
        )

    driver_df = pd.DataFrame(records)
    if driver_df.empty:
        return driver_df

    driver_df["driver_eligible"] = driver_df["delta_mean"] > 0
    driver_df["driver_burden_score"] = np.where(
        driver_df["driver_eligible"],
        driver_df["excess_high_shift_share"],
        np.nan,
    )

    driver_df = driver_df.sort_values(
        ["driver_eligible", "driver_burden_score", "delta_mean", "excess_positive_shift_mass_share", "n_case_cells"],
        ascending=[False, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    driver_df["driver_rank"] = np.arange(1, driver_df.shape[0] + 1, dtype=int)
    ordered_cols = ["driver_rank", group_key] + [col for col in driver_df.columns if col not in {"driver_rank", group_key}]
    return driver_df[ordered_cols]

def build_state_shifts(
    mapped_obs: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    state_key = str(cfg["mapping"]["primary_label_key"])
    support_key = f"{state_key}_support"
    perturbation_key = str(cfg["design"]["perturbation_key"])
    donor_key = str(cfg["design"]["donor_id_key"])
    control_label = str(cfg["design"]["control_label"])
    panel = cfg["state_summary"]
    metric_key = "healthy_timepoint_exceedance"
    high_key = "healthy_timepoint_q90_flag"
    required = {
        state_key, support_key, perturbation_key, donor_key, "timepoint_label", metric_key, high_key,
    }
    missing = sorted(required - set(mapped_obs.columns))
    if missing:
        raise KeyError(f"Mapped query is missing State summary fields: {missing}")

    work = mapped_obs.copy()
    work = work.loc[
        work[support_key].astype(float).ge(float(panel["label_support_min"]))
        & work[state_key].notna()
        & work[donor_key].notna()
        & work[perturbation_key].notna()
        & work["timepoint_label"].notna()
    ].copy()
    work[state_key] = work[state_key].astype(str)
    work[donor_key] = work[donor_key].astype(str)
    work[perturbation_key] = work[perturbation_key].astype(str)

    def balanced_high_pool(frame: pd.DataFrame, states: Sequence[str]) -> tuple[pd.Series, pd.Series, int]:


        if frame.empty or not states:
            return (
                pd.Series(np.nan, index=list(states), dtype=float),
                pd.Series(0, index=list(states), dtype=int),
                0,
            )
        unit_keys = [donor_key, perturbation_key]
        highs = (
            frame.groupby(unit_keys + [state_key], observed=True)[high_key].sum().unstack(state_key, fill_value=0)
        ).reindex(columns=list(states), fill_value=0)
        valid_units = highs.sum(axis=1).gt(0)
        if not bool(valid_units.any()):
            return pd.Series(np.nan, index=list(states), dtype=float), highs.sum(axis=0), 0
        shares = highs.loc[valid_units].div(highs.loc[valid_units].sum(axis=1), axis=0).mean(axis=0)
        return shares, highs.sum(axis=0), int(valid_units.sum())

    records: list[dict[str, Any]] = []
    for timepoint, timepoint_df in work.groupby("timepoint_label", observed=True, sort=True):
        case_df = timepoint_df.loc[~timepoint_df[perturbation_key].eq(control_label)].copy()
        control_df = timepoint_df.loc[timepoint_df[perturbation_key].eq(control_label)].copy()
        states = sorted(case_df[state_key].unique().tolist())
        if not states:
            continue
        metric_timepoint = timepoint_df.loc[timepoint_df[metric_key].notna()].copy()
        unit_keys = [donor_key, perturbation_key]
        unit_sizes = metric_timepoint.groupby(unit_keys, observed=True)[metric_key].transform("size")
        metric_timepoint = metric_timepoint.loc[
            unit_sizes.ge(int(cfg["design"].get("min_cells_per_sample_unit", 20)))
        ].copy()
        metric_case = metric_timepoint.loc[
            ~metric_timepoint[perturbation_key].eq(control_label)
        ].copy()
        metric_control = metric_timepoint.loc[
            metric_timepoint[perturbation_key].eq(control_label)
        ].copy()
        case_state_units = (
            metric_case
            .groupby([donor_key, perturbation_key, state_key], observed=True)
            .agg(
                sample_unit_n_cells=(state_key, "size"),
                sample_unit_mean_exceedance=(metric_key, "mean"),
                sample_unit_median_exceedance=(metric_key, "median"),
            )
            .reset_index()
        )
        control_units = (
            metric_control.groupby(donor_key, observed=True)
            .agg(
                sample_unit_n_cells=(state_key, "size"),
                sample_unit_mean_exceedance=(metric_key, "mean"),
            )
            .reset_index()
        )
        control_center = (
            float(control_units["sample_unit_mean_exceedance"].mean())
            if not control_units.empty else float("nan")
        )

        prelim: dict[str, dict[str, Any]] = {}
        eligible_states: list[str] = []
        for state in states:
            state_case = case_df.loc[case_df[state_key].eq(state)]
            state_control = control_df.loc[control_df[state_key].eq(state)]
            state_units = case_state_units.loc[case_state_units[state_key].eq(state)]
            eligible_case_units = state_units.loc[
                state_units["sample_unit_n_cells"].ge(int(panel["min_cells_per_sample_unit_state"]))
            ]
            support_median = float(state_case[support_key].median()) if not state_case.empty else np.nan
            reasons: list[str] = []
            if int(state_case.shape[0]) < int(panel["min_case_cells"]):
                reasons.append("case_cells_below_min")
            if int(state_case[donor_key].nunique()) < int(panel["min_case_donors"]):
                reasons.append("case_donors_below_min")
            if int(eligible_case_units.shape[0]) < int(panel["min_case_sample_units"]):
                reasons.append("case_sample_units_below_min")
            if not np.isfinite(support_median) or support_median < float(panel["min_state_support_median"]):
                reasons.append("state_support_median_below_min")
            if eligible_case_units.empty:
                reasons.append("case_metric_unavailable")
            if not np.isfinite(control_center):
                reasons.append("timepoint_control_baseline_unavailable")
            prelim[state] = {
                "state_case": state_case,
                "state_control": state_control,
                "case_units": state_units,
                "eligible_case_units": eligible_case_units,
                "support_median": support_median,
                "reasons": reasons,
            }
            if not reasons:
                eligible_states.append(state)

        share_case = metric_case.loc[metric_case[state_key].isin(eligible_states)]
        share_control = metric_control.loc[metric_control[state_key].isin(eligible_states)]
        case_shares, case_high_cells, n_valid_case_units = balanced_high_pool(share_case, eligible_states)
        control_shares, control_high_cells, n_valid_control_units = balanced_high_pool(
            share_control, eligible_states
        )
        shares_available = bool(eligible_states) and n_valid_case_units > 0 and n_valid_control_units > 0

        for state in states:
            info = prelim[state]
            state_case = info["state_case"]
            state_control = info["state_control"]
            case_units = info["case_units"]
            eligible_case_units = info["eligible_case_units"]
            reasons = list(info["reasons"])
            eligible = state in eligible_states and shares_available
            if state in eligible_states and not shares_available:
                reasons.append("high_pool_unavailable")
            if eligible:
                case_mean = float(eligible_case_units["sample_unit_mean_exceedance"].mean())
                case_median = float(eligible_case_units["sample_unit_median_exceedance"].median())
                mean_exceedance_shift = case_mean - control_center
                case_share = float(case_shares[state])
                control_share = float(control_shares[state])
                high_exceedance_share_shift = case_share - control_share
                case_high = int(case_high_cells[state])
                control_high = int(control_high_cells[state])
            else:
                case_mean = case_median = float("nan")
                mean_exceedance_shift = case_share = control_share = high_exceedance_share_shift = float("nan")
                case_high = int(state_case[high_key].sum())
                control_high = int(state_control[high_key].sum())
            records.append(
                {
                    "timepoint_label": str(timepoint),
                    "level3": state,
                    "n_case_cells": int(state_case.shape[0]),
                    "n_control_cells": int(state_control.shape[0]),
                    "n_case_donors": int(state_case[donor_key].nunique()),
                    "n_control_donors": int(state_control[donor_key].nunique()),
                    "n_case_perturbations": int(state_case[perturbation_key].nunique()),
                    "n_case_sample_units": int(case_units.shape[0]),
                    "n_eligible_case_sample_units": int(eligible_case_units.shape[0]),
                    "n_timepoint_control_sample_units": int(control_units.shape[0]),
                    "n_valid_case_high_pool_sample_units": n_valid_case_units if eligible else 0,
                    "n_valid_control_high_pool_sample_units": n_valid_control_units if eligible else 0,
                    "case_sample_unit_mean_exceedance": case_mean,
                    "case_sample_unit_median_exceedance": case_median,
                    "timepoint_control_sample_unit_mean_exceedance": control_center,
                    "mean_exceedance_shift": mean_exceedance_shift,
                    "case_high_cells": case_high,
                    "control_high_cells": control_high,
                    "case_high_pool_share_balanced": case_share,
                    "control_high_pool_share_balanced": control_share,
                    "high_exceedance_share_shift": high_exceedance_share_shift,
                    "support_mean": float(state_case[support_key].mean()) if not state_case.empty else np.nan,
                    "support_median": info["support_median"],
                    "driver_eligible": bool(eligible),
                    "exclusion_reason": ";".join(reasons),
                }
            )

    result = pd.DataFrame(records)
    if result.empty:
        return result
    ranked: list[pd.DataFrame] = []
    for _, frame in result.groupby("timepoint_label", observed=True, sort=True):
        frame = frame.copy()
        frame["driver_rank"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
        ranked_index = frame.loc[frame["driver_eligible"] & frame["mean_exceedance_shift"].gt(0)].sort_values(
            ["high_exceedance_share_shift", "mean_exceedance_shift"], ascending=[False, False]
        ).index
        frame.loc[ranked_index, "driver_rank"] = np.arange(1, len(ranked_index) + 1, dtype=int)
        ranked.append(frame.sort_values(["driver_rank", "level3"], na_position="last"))
    result = pd.concat(ranked, ignore_index=True)
    columns = [
        "timepoint_label", "level3",
        "n_case_cells", "n_control_cells", "n_case_donors", "n_control_donors",
        "n_case_perturbations", "n_case_sample_units", "n_eligible_case_sample_units",
        "n_timepoint_control_sample_units", "n_valid_case_high_pool_sample_units",
        "n_valid_control_high_pool_sample_units", "case_sample_unit_mean_exceedance",
        "case_sample_unit_median_exceedance", "timepoint_control_sample_unit_mean_exceedance",
        "mean_exceedance_shift", "case_high_cells", "control_high_cells",
        "case_high_pool_share_balanced", "control_high_pool_share_balanced", "high_exceedance_share_shift",
        "support_mean", "support_median", "driver_eligible", "exclusion_reason", "driver_rank",
    ]
    return result[columns]

def build_perturbation_exceedance_summary(
    mapped_obs: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    perturbation_key = str(cfg["design"]["perturbation_key"])
    donor_key = str(cfg["design"]["donor_id_key"])
    state_key = str(cfg["mapping"]["primary_label_key"])
    support_key = f"{state_key}_support"
    all_work = mapped_obs.copy()
    all_work["support_pass"] = all_work[support_key].astype(float).ge(
        float(cfg["state_summary"]["label_support_min"])
    )
    base = (
        all_work.groupby(["timepoint_label", perturbation_key], observed=True)
        .agg(support_pass_fraction=("support_pass", "mean"))
        .reset_index()
    )
    work = _distribution_cells(all_work, cfg)
    cell_summary = (
        work.groupby(["timepoint_label", perturbation_key], observed=True)
        .agg(
            n_cells=(perturbation_key, "size"),
            n_donors=(donor_key, "nunique"),
            cell_mean_exceedance=("healthy_timepoint_exceedance", "mean"),
            cell_median_exceedance=("healthy_timepoint_exceedance", "median"),
            mean_healthy_distance=("healthy_neighbor_mean_distance", "mean"),
        )
        .reset_index()
    )
    donor_summary = (
        work.groupby(["timepoint_label", perturbation_key, donor_key], observed=True)
        .agg(
            donor_mean_exceedance=("healthy_timepoint_exceedance", "mean"),
            donor_high_fraction=("healthy_timepoint_q90_flag", "mean"),
        )
        .reset_index()
    )
    donor_balanced = (
        donor_summary.groupby(["timepoint_label", perturbation_key], observed=True)
        .agg(
            donor_balanced_mean_exceedance=("donor_mean_exceedance", "mean"),
            donor_balanced_high_fraction=("donor_high_fraction", "mean"),
        )
        .reset_index()
    )
    result = (
        cell_summary.merge(donor_balanced, on=["timepoint_label", perturbation_key], how="left")
        .merge(base, on=["timepoint_label", perturbation_key], how="left")
    )
    control_label = str(cfg["design"]["control_label"])
    paired_records: list[dict[str, Any]] = []
    for timepoint, frame in donor_summary.groupby("timepoint_label", observed=True, sort=True):
        control = frame.loc[
            frame[perturbation_key].astype(str).eq(control_label),
            [donor_key, "donor_mean_exceedance"],
        ].rename(columns={"donor_mean_exceedance": "control_donor_mean"})
        for perturbation, case in frame.groupby(perturbation_key, observed=True, sort=True):
            perturbation = str(perturbation)
            if perturbation == control_label:
                delta = 0.0
                n_paired = int(case[donor_key].nunique())
            else:
                paired = case[[donor_key, "donor_mean_exceedance"]].merge(control, on=donor_key, how="inner")
                n_paired = int(paired.shape[0])
                delta = (
                    float((paired["donor_mean_exceedance"] - paired["control_donor_mean"]).mean())
                    if n_paired
                    else float("nan")
                )
            paired_records.append(
                {
                    "timepoint_label": str(timepoint),
                    perturbation_key: perturbation,
                    "paired_no_stim_delta_mean": delta,
                    "n_paired_donors": n_paired,
                }
            )
    result = result.merge(pd.DataFrame(paired_records), on=["timepoint_label", perturbation_key], how="left")
    result = result.rename(columns={perturbation_key: "perturbation"})
    result = result.sort_values(
        ["timepoint_label", "paired_no_stim_delta_mean", "donor_balanced_mean_exceedance", "donor_balanced_high_fraction"],
        ascending=[True, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    result["shift_rank"] = result.groupby("timepoint_label", observed=True).cumcount() + 1
    return result[
        [
            "timepoint_label", "perturbation", "n_cells", "n_donors",
            "cell_mean_exceedance", "cell_median_exceedance",
            "donor_balanced_mean_exceedance", "donor_balanced_high_fraction",
            "paired_no_stim_delta_mean", "n_paired_donors", "mean_healthy_distance",
            "support_pass_fraction", "shift_rank",
        ]
    ]

def _distribution_cells(mapped_obs: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    perturbation_key = str(cfg["design"]["perturbation_key"])
    donor_key = str(cfg["design"]["donor_id_key"])
    metric_key = "healthy_timepoint_exceedance"
    required = {"timepoint_label", perturbation_key, donor_key, metric_key}
    missing = sorted(required - set(mapped_obs.columns))
    if missing:
        raise KeyError(f"Mapped query is missing distribution fields: {missing}")
    work = mapped_obs.loc[
        mapped_obs["timepoint_label"].notna()
        & mapped_obs[perturbation_key].notna()
        & mapped_obs[donor_key].notna()
        & mapped_obs[metric_key].notna()
    ].copy()
    unit_keys = ["timepoint_label", perturbation_key, donor_key]
    unit_sizes = work.groupby(unit_keys, observed=True)[metric_key].transform("size")
    return work.loc[
        unit_sizes.ge(int(cfg["design"].get("min_cells_per_sample_unit", 20)))
    ].copy()

def prepare_design_obs(query_obs: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    design_cfg = cfg["design"]
    donor_key = str(design_cfg["donor_id_key"])
    perturbation_key = str(design_cfg["perturbation_key"])
    control_label = str(design_cfg["control_label"])
    timepoint_map = {str(k): str(v) for k, v in dict(design_cfg["timepoint_map"]).items()}

    missing = [key for key in [donor_key, perturbation_key] if key not in query_obs.columns]
    if missing:
        raise KeyError(f"Query obs is missing design columns: {missing}")

    design = query_obs.copy()
    design.index = design.index.astype(str)
    design[donor_key] = _clean_string_series(design[donor_key]).to_numpy()
    design[perturbation_key] = _clean_string_series(design[perturbation_key]).to_numpy()

    donor_compact = design[donor_key].str.replace(r"[\s_\-]+", "", regex=True)
    strict = donor_compact.str.extract(r"^(?P<donor>[A-Za-z]+\d+)(?P<timepoint>D\d+)$")
    fallback = donor_compact.str.extract(r"(?P<donor>[A-Za-z]+\d+).*?(?P<timepoint>D\d+)")
    design["donor"] = strict["donor"].fillna(fallback["donor"]).fillna("")
    design["timepoint"] = strict["timepoint"].fillna(fallback["timepoint"]).fillna("")
    design["timepoint_label"] = design["timepoint"].map(timepoint_map).fillna("")
    design["donor_timepoint_parse_ok"] = design["donor"].ne("") & design["timepoint_label"].ne("")
    design["is_control"] = design[perturbation_key].eq(control_label)
    design["analysis_group"] = design[perturbation_key] + " | " + design["timepoint_label"]

    sample_fields = [str(value) for value in design_cfg.get("sample_unit_fields", [donor_key, perturbation_key])]
    missing_sample_fields = [field for field in sample_fields if field not in design.columns]
    if missing_sample_fields:
        raise KeyError(f"sample_unit_fields are missing from query obs: {missing_sample_fields}")
    design["sample_unit"] = design[sample_fields].astype(str).agg("__".join, axis=1)
    return design

def _clean_string_series(values: Iterable[object]) -> pd.Series:
    series = pd.Series(values, copy=False)
    if isinstance(series.dtype, pd.CategoricalDtype):
        series = series.astype(object)
    return series.where(pd.notna(series), "").astype(str).str.strip()


def standardize(table, profile):
    table = table.copy()
    x, y = ('delta_mean', 'excess_high_shift_share') if profile == 'matched_control' else ('mean_exceedance_shift', 'high_exceedance_share_shift')
    table['mean_exceedance_shift'] = table[x]
    table['high_exceedance_share_shift'] = table[y]
    return table


def build_matched_control_tables(cells, cfg):
    tables = {}
    for level, key in [('level2', str(cfg['broad_group_key'])), ('level3', str(cfg['fine_group_key']))]:
        for case in cfg['analysis_case_groups']:
            t = build_shift_driver_table(cells, cfg, str(case), key)
            if not t.empty:
                tables[(level, str(case))] = standardize(t, 'matched_control')
    return tables


def design_perturbation_control(obs, cfg):
    from .inputs import check_keys
    design = cfg['design']
    check_keys(obs, [design['donor_id_key'], design['perturbation_key']])
    time_key = design.get('timepoint_key', 'timepoint_label')
    if time_key in obs:
        result = obs.copy()
        check_keys(result, [time_key])
        result['timepoint_label'] = result[time_key].astype(str)
        result['is_control'] = result[design['perturbation_key']].astype(str).eq(design['control_label'])
        result['sample_unit'] = result[[design['donor_id_key'], design['perturbation_key'], 'timepoint_label']].astype(str).agg('__'.join, axis=1)
        return result
    result = prepare_design_obs(obs, cfg)
    if not result['donor_timepoint_parse_ok'].all():
        raise ValueError('Cannot infer timepoint; provide design.timepoint_key and explicit metadata')
    return result


def perturbation_control_unit_metrics(cells, cfg):
    donor = cfg['design']['donor_id_key']; treatment = cfg['design']['perturbation_key']
    state = cfg['mapping']['primary_label_key']; metric = 'healthy_timepoint_exceedance'
    work = cells.loc[cells[metric].notna()].copy()
    keys = ['timepoint_label', donor, treatment]
    sizes = work.groupby(keys, observed=True)[metric].transform('size')
    work['unit_eligible'] = sizes.ge(cfg['design']['min_cells_per_sample_unit'])
    result = work.groupby(keys + [state], observed=True).agg(n_cells=(metric, 'size'), mean_exceedance=(metric, 'mean'), median_exceedance=(metric, 'median'), high_fraction=('healthy_timepoint_q90_flag', 'mean'), high_cells=('healthy_timepoint_q90_flag', 'sum'), unit_eligible=('unit_eligible', 'first')).reset_index()
    result['state_unit_eligible'] = result['unit_eligible'] & result['n_cells'].ge(cfg['state_summary']['min_cells_per_sample_unit_state'])
    return result


def perturbation_control_by_perturbation(cells, cfg):
    from copy import deepcopy
    config = deepcopy(cfg)
    config['state_summary'].update({'min_case_cells': 20, 'min_case_donors': 2, 'min_cells_per_sample_unit_state': 5, 'min_case_sample_units': 2, 'min_state_support_median': .75})
    config['state_summary'].update(cfg.get('by_perturbation', {}))
    key = cfg['design']['perturbation_key']; control = cfg['design']['control_label']
    frames = []
    for value in sorted(set(cells[key].astype(str)) - {control}):
        subset = cells.loc[cells[key].astype(str).isin([control, value])]
        result = build_state_shifts(subset, config)
        if not result.empty:
            result.insert(1, 'perturbation', value)
            frames.append(result)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
