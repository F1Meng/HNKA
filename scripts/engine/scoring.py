from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def ecdf(values, reference_values, method="right"):
    values = np.asarray(values, dtype=float)
    reference = np.sort(np.asarray(reference_values, dtype=float))
    if reference.size == 0 or not np.isfinite(reference).all():
        raise ValueError("ECDF needs a nonempty finite reference")
    if not np.isfinite(values).all():
        raise ValueError("ECDF query contains nonfinite distances")
    right = np.searchsorted(reference, values, side="right")
    if method == "right":
        return right / float(reference.size)
    if method == "midrank":
        left = np.searchsorted(reference, values, side="left")
        return (left + right) / (2.0 * float(reference.size))
    raise ValueError(f"Unknown ECDF method: {method}")

def _mean_knn_distance(reference_latent: np.ndarray, query_latent: np.ndarray, k: int) -> np.ndarray:
    if reference_latent.shape[0] == 0 or query_latent.shape[0] == 0:
        return np.full(query_latent.shape[0], np.nan, dtype=float)
    n_neighbors = min(int(k), reference_latent.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(reference_latent)
    distances, _ = nn.kneighbors(query_latent)
    return distances.mean(axis=1)

def build_query_normal_baselines(
    scope_df: pd.DataFrame,
    cfg: Dict[str, object],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    analysis_key = str(cfg["analysis_axis_key"])
    control_label = str(cfg["analysis_control_label"])
    broad_key = str(cfg["broad_group_key"])
    organ_key = str(cfg["organ_key"])
    donor_key = str(cfg["donor_key"])
    raw_key = str(cfg["raw_distance_key"])
    q = float(cfg["high_shift_quantile"])

    normal_df = scope_df.loc[scope_df[analysis_key] == control_label].copy()
    if normal_df.empty:
        raise ValueError("No query Normal cells available to build baseline distribution")

    baseline_l2_org_df = (
        normal_df.groupby([broad_key, organ_key], observed=True)
        .agg(
            baseline_normal_median=(raw_key, "median"),
            baseline_normal_q90=(raw_key, lambda x: float(np.quantile(x, q))),
            baseline_normal_iqr=(raw_key, lambda x: float(np.quantile(x, 0.75) - np.quantile(x, 0.25))),
            baseline_normal_cells=(raw_key, "size"),
            baseline_normal_donors=(donor_key, "nunique"),
        )
        .reset_index()
        .rename(columns={broad_key: "baseline_broad_group", organ_key: "baseline_organ"})
    )
    baseline_l2_df = (
        normal_df.groupby([broad_key], observed=True)
        .agg(
            baseline_normal_median=(raw_key, "median"),
            baseline_normal_q90=(raw_key, lambda x: float(np.quantile(x, q))),
            baseline_normal_iqr=(raw_key, lambda x: float(np.quantile(x, 0.75) - np.quantile(x, 0.25))),
            baseline_normal_cells=(raw_key, "size"),
            baseline_normal_donors=(donor_key, "nunique"),
        )
        .reset_index()
        .rename(columns={broad_key: "baseline_broad_group"})
    )
    baseline_global_df = pd.DataFrame(
        [
            {
                "baseline_normal_median": float(normal_df[raw_key].median()),
                "baseline_normal_q90": float(np.quantile(normal_df[raw_key].to_numpy(dtype=float), q)),
                "baseline_normal_iqr": float(
                    np.quantile(normal_df[raw_key].to_numpy(dtype=float), 0.75)
                    - np.quantile(normal_df[raw_key].to_numpy(dtype=float), 0.25)
                ),
                "baseline_normal_cells": int(normal_df.shape[0]),
                "baseline_normal_donors": int(normal_df[donor_key].nunique()),
            }
        ]
    )
    return baseline_l2_org_df, baseline_l2_df, baseline_global_df

def assign_query_normal_baseline(
    scope_df: pd.DataFrame,
    baseline_l2_org_df: pd.DataFrame,
    baseline_l2_df: pd.DataFrame,
    baseline_global_df: pd.DataFrame,
    cfg: Dict[str, object],
) -> pd.DataFrame:
    broad_key = str(cfg["broad_group_key"])
    organ_key = str(cfg["organ_key"])
    min_donors = int(cfg["baseline_min_normal_donors"])
    min_cells = int(cfg["baseline_min_normal_cells"])

    result_df = scope_df.copy()
    result_df = result_df.merge(
        baseline_l2_org_df,
        left_on=[broad_key, organ_key],
        right_on=["baseline_broad_group", "baseline_organ"],
        how="left",
        suffixes=("", "_l2org"),
    )

    for column in [
        "baseline_normal_median",
        "baseline_normal_q90",
        "baseline_normal_iqr",
        "baseline_normal_cells",
        "baseline_normal_donors",
    ]:
        result_df = result_df.rename(columns={column: f"{column}_l2org"})

    result_df = result_df.merge(
        baseline_l2_df,
        left_on=broad_key,
        right_on="baseline_broad_group",
        how="left",
        suffixes=("", "_l2"),
    )
    for column in [
        "baseline_normal_median",
        "baseline_normal_q90",
        "baseline_normal_iqr",
        "baseline_normal_cells",
        "baseline_normal_donors",
    ]:
        result_df = result_df.rename(columns={column: f"{column}_l2"})

    baseline_global = baseline_global_df.iloc[0].to_dict()
    use_l2org = (
        result_df["baseline_normal_cells_l2org"].fillna(0).astype(float) >= min_cells
    ) & (
        result_df["baseline_normal_donors_l2org"].fillna(0).astype(float) >= min_donors
    )

    for column in ["median", "q90", "iqr", "cells", "donors"]:
        l2org_col = f"baseline_normal_{column}_l2org"
        l2_col = f"baseline_normal_{column}_l2"
        target_col = f"baseline_normal_{column}"
        result_df[target_col] = np.where(
            use_l2org,
            result_df[l2org_col],
            result_df[l2_col],
        )
        if pd.notna(baseline_global.get(f"baseline_normal_{column}")):
            result_df[target_col] = result_df[target_col].fillna(baseline_global[f"baseline_normal_{column}"])

    result_df["baseline_scope"] = np.where(use_l2org, "level2+organ", "level2")
    result_df.loc[result_df["baseline_normal_cells_l2"].isna(), "baseline_scope"] = "global_normal"
    drop_cols = [
        "baseline_broad_group",
        "baseline_organ",
        "baseline_broad_group_l2",
    ]
    result_df = result_df.drop(columns=[col for col in drop_cols if col in result_df.columns])
    if "obs_name" in result_df.columns:
        result_df = result_df.set_index("obs_name", drop=False)
    return result_df

def _sorted_float_values(series: pd.Series) -> np.ndarray:
    values = series.dropna().to_numpy(dtype=float)
    if values.size == 0:
        return np.array([], dtype=float)
    return np.sort(values)

def _empirical_percentile(values: np.ndarray, reference_values: np.ndarray) -> np.ndarray:
    if reference_values.size == 0:
        return np.full(values.shape[0], np.nan, dtype=float)
    return ecdf(values, reference_values, "right")

def assign_normal_exceedance_scores(
    scope_df: pd.DataFrame,
    cfg: Dict[str, object],
) -> pd.DataFrame:
    analysis_key = str(cfg["analysis_axis_key"])
    control_label = str(cfg["analysis_control_label"])
    broad_key = str(cfg["broad_group_key"])
    organ_key = str(cfg["organ_key"])
    raw_key = str(cfg["raw_distance_key"])
    exceedance_key = str(cfg["atlas_exceedance_key"])
    residual_key = str(cfg["atlas_exceedance_residual_key"])
    high_flag_key = str(cfg["atlas_high_exceedance_flag_key"])

    normal_df = scope_df.loc[scope_df[analysis_key].eq(control_label)].copy()
    if normal_df.empty:
        raise ValueError("No matched Normal cells available to build exceedance scores")

    l2org_lookup = {
        (str(broad_value), str(organ_value)): _sorted_float_values(group_df[raw_key])
        for (broad_value, organ_value), group_df in normal_df.groupby([broad_key, organ_key], observed=True)
    }
    l2_lookup = {
        str(broad_value): _sorted_float_values(group_df[raw_key])
        for broad_value, group_df in normal_df.groupby(broad_key, observed=True)
    }
    global_lookup = _sorted_float_values(normal_df[raw_key])

    result_df = scope_df.copy()
    exceedance_scores = np.full(result_df.shape[0], np.nan, dtype=float)
    index_positions = pd.Series(np.arange(result_df.shape[0], dtype=int), index=result_df.index)

    mask_l2org = result_df["baseline_scope"].astype(str).eq("level2+organ")
    if mask_l2org.any():
        for (broad_value, organ_value), idx in result_df.loc[mask_l2org].groupby([broad_key, organ_key], observed=True).groups.items():
            ref_values = l2org_lookup.get((str(broad_value), str(organ_value)), np.array([], dtype=float))
            raw_values = result_df.loc[idx, raw_key].to_numpy(dtype=float)
            exceedance_scores[index_positions.loc[idx].to_numpy(dtype=int)] = _empirical_percentile(raw_values, ref_values)

    mask_l2 = result_df["baseline_scope"].astype(str).eq("level2")
    if mask_l2.any():
        for broad_value, idx in result_df.loc[mask_l2].groupby(broad_key, observed=True).groups.items():
            ref_values = l2_lookup.get(str(broad_value), np.array([], dtype=float))
            raw_values = result_df.loc[idx, raw_key].to_numpy(dtype=float)
            exceedance_scores[index_positions.loc[idx].to_numpy(dtype=int)] = _empirical_percentile(raw_values, ref_values)

    mask_global = result_df["baseline_scope"].astype(str).eq("global_normal")
    if mask_global.any():
        global_idx = result_df.index[mask_global]
        raw_values = result_df.loc[global_idx, raw_key].to_numpy(dtype=float)
        exceedance_scores[index_positions.loc[global_idx].to_numpy(dtype=int)] = _empirical_percentile(raw_values, global_lookup)

    result_df[exceedance_key] = exceedance_scores
    result_df[residual_key] = result_df[exceedance_key] - 0.5
    result_df[high_flag_key] = (result_df[exceedance_key] >= 0.90).astype(int)
    return result_df

def _build_neighbor_index(reference: np.ndarray, n_neighbors: int, seed: int, backend: str = "pynndescent") -> tuple[Any, str]:
    reference = np.asarray(reference, dtype=np.float32)
    if reference.shape[0] == 0:
        raise ValueError("Cannot build a neighbor index from an empty reference")
    requested = min(max(int(n_neighbors), 1), int(reference.shape[0]))
    if backend == "sklearn":
        index = NearestNeighbors(n_neighbors=requested, metric="euclidean", n_jobs=-1)
        index.fit(reference)
        return index, "sklearn"
    if backend != "pynndescent":
        raise ValueError(f"Unsupported neighbor backend: {backend}")
    try:
        from pynndescent import NNDescent

        index = NNDescent(
            reference,
            n_neighbors=min(max(requested, 15), int(reference.shape[0])),
            metric="euclidean",
            random_state=int(seed),
            low_memory=True,
            n_jobs=-1,
        )
        index.prepare()
        return index, "pynndescent"
    except ImportError:
        raise ImportError("pynndescent is required by this configuration; refusing silent backend change")

def _query_neighbor_index(index: Any, query: np.ndarray, n_neighbors: int, backend: str) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(query, dtype=np.float32)
    if backend == "pynndescent":
        indices, distances = index.query(query, k=int(n_neighbors))
    else:
        distances, indices = index.kneighbors(query, n_neighbors=int(n_neighbors))
    return np.asarray(indices), np.asarray(distances)

def empirical_percentile(values: np.ndarray, reference_values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    reference = np.sort(np.asarray(reference_values, dtype=float))
    if reference.size == 0:
        raise ValueError("Empirical percentile requires a non-empty reference")
    result = ecdf(values, reference, "midrank")

    return result

def assign_healthy_timepoint_exceedance(
    mapped_obs: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    state_key = str(cfg["mapping"]["primary_label_key"])
    support_key = f"{state_key}_support"
    perturbation_key = str(cfg["design"]["perturbation_key"])
    donor_key = str(cfg["design"]["donor_id_key"])
    control_label = str(cfg["design"]["control_label"])
    panel = cfg["state_summary"]
    reference_cfg = cfg["healthy_timepoint_reference"]
    distance_key = "healthy_neighbor_mean_distance"
    exceedance_key = "healthy_timepoint_exceedance"
    high_key = "healthy_timepoint_q90_flag"
    required = {
        state_key, support_key, perturbation_key, donor_key, "timepoint_label", distance_key,
    }
    missing = sorted(required - set(mapped_obs.columns))
    if missing:
        raise KeyError(f"Mapped query is missing healthy-timepoint fields: {missing}")

    result = mapped_obs.copy()
    result[exceedance_key] = np.nan
    result[high_key] = False
    support_pass = result[support_key].astype(float).ge(float(panel["label_support_min"]))
    valid = (
        support_pass
        & result[state_key].notna()
        & result["timepoint_label"].notna()
        & result[distance_key].notna()
    )
    grouped = result.loc[valid].groupby("timepoint_label", observed=True, sort=True)
    for timepoint, group in grouped:
        control = group.loc[group[perturbation_key].astype(str).eq(control_label)]
        n_control_cells = int(control.shape[0])
        n_control_donors = int(control[donor_key].nunique())
        eligible = (
            n_control_cells >= int(reference_cfg["min_control_cells"])
            and n_control_donors >= int(reference_cfg["min_control_donors"])
        )
        if eligible:
            reference = control[distance_key].to_numpy(dtype=float)
            values = group[distance_key].to_numpy(dtype=float)
            exceedance = empirical_percentile(values, reference)
            result.loc[group.index, exceedance_key] = exceedance
            result.loc[group.index, high_key] = exceedance >= 0.90
    return result


def calibrate_matched_control(scope, cfg, controls=None):
    reference = scope if controls is None else controls
    b1, b2, bg = build_query_normal_baselines(reference, cfg)
    result = assign_query_normal_baseline(scope, b1, b2, bg, cfg)
    raw, shift, robust = (str(cfg[k]) for k in ['raw_distance_key', 'atlas_shift_key', 'atlas_shift_robust_key'])
    result[shift] = result[raw] - result['baseline_normal_median']
    result[robust] = result[shift] / result['baseline_normal_iqr'].replace(0.0, np.nan)
    result[str(cfg['atlas_high_flag_key'])] = (result[raw] > result['baseline_normal_q90']).astype(int)
    if controls is None:
        result = assign_normal_exceedance_scores(result, cfg)
    else:

        normal = controls.copy()
        normal.index = pd.Index(['__calibration__' + str(i) for i in range(len(normal))])
        original_index = result.index.copy()
        temporary = result.copy()
        temporary.index = pd.Index(['__query__' + str(i) for i in range(len(result))])
        if temporary[str(cfg['analysis_axis_key'])].eq(str(cfg['analysis_control_label'])).any():
            raise ValueError('Expected query labels distinct from the reference control label')
        joined = pd.concat([normal, temporary], axis=0)
        scored = assign_normal_exceedance_scores(joined, cfg).loc[temporary.index]
        for key in ['atlas_exceedance_key', 'atlas_exceedance_residual_key', 'atlas_high_exceedance_flag_key']:
            result[str(cfg[key])] = scored[str(cfg[key])].to_numpy()
        result.index = original_index
    result['is_normal'] = result[str(cfg['analysis_axis_key'])].eq(str(cfg['analysis_control_label']))
    result['is_disease'] = ~result['is_normal']
    return result


def matched_control_scope(adata, cfg):
    from .inputs import build_obs_table
    obs = build_obs_table(adata, cfg)
    axis = str(cfg['analysis_axis_key'])
    mask = obs[str(cfg['query_key'])].eq(str(cfg['query_value']))
    mask &= obs[axis].isin([str(cfg['analysis_control_label'])] + list(cfg['analysis_case_groups']))
    for key, values in cfg.get('cohort_filters', {}).items():
        mask &= obs[key].isin(values)
    mask &= ~obs[str(cfg['broad_group_key'])].isin(cfg['exclude_broad_groups'])
    mask &= ~obs[str(cfg['fine_group_key'])].isin(cfg['exclude_fine_groups'])
    result = obs.loc[mask].copy()
    if result.empty:
        raise ValueError('No analysis cells pass the configured scope')
    return result


def source_geometry(atlas, cfg, profile):
    if profile == 'matched_control':
        from .inputs import normalize_string_series
        source = normalize_string_series(atlas.obs[str(cfg['query_key'])]).eq(str(cfg['source_value'])).to_numpy(dtype=bool)
        key = str(cfg['latent_key'])
        dtype = float
    else:
        rule = cfg['healthy_reference']
        values = atlas.obs[str(rule['query_key'])]
        def zero(v):
            if pd.isna(v):
                return False
            return str(v).strip().lower() in {'0', '0.0', 'false'}
        source = values.map(zero).to_numpy() if zero(rule.get('query_value', 0)) else values.astype(str).eq(str(rule['query_value'])).to_numpy()
        key = str(cfg['mapping']['latent_key'])
        dtype = np.float32
    if not source.any():
        raise ValueError('No source reference cells; external fallback is disabled')
    return np.asarray(atlas.obsm[key][source], dtype=dtype), atlas.obs_names[source].astype(str).to_numpy()


def score_perturbation_control_geometry(source, query_latent, cfg):
    k = min(int(cfg['healthy_reference']['k']), len(source) - 1)
    if k < 1:
        raise ValueError('Healthy reference needs at least two cells')
    index, backend = _build_neighbor_index(source, k + 1, int(cfg['mapping']['random_seed']), cfg['neighbor_backend'])
    _, distances = _query_neighbor_index(index, query_latent, k, backend)
    distances = np.asarray(distances, dtype=float).mean(axis=1)
    return pd.DataFrame({'healthy_neighbor_mean_distance': distances})
