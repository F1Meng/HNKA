from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import json

def normalize_string_series(series: pd.Series, fill_value: str = "Unknown") -> pd.Series:
    normalized = pd.Series(series, copy=True).astype("string").str.strip()
    normalized = normalized.replace(
        {
            "<NA>": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
            "None": pd.NA,
            "": pd.NA,
        }
    )
    return normalized.fillna(fill_value)

def normalize_condition_series(series: pd.Series, control_label: str) -> pd.Series:
    normalized = normalize_string_series(series)
    control_label = str(control_label).strip()
    if control_label == "Normal":
        normalized = normalized.replace({"Healthy": "Normal", "Healthy (Normal)": "Normal"})
    elif control_label == "Healthy (Normal)":
        normalized = normalized.replace({"Healthy": "Healthy (Normal)", "Normal": "Healthy (Normal)"})
    return normalized.astype(str)

def normalize_filter_dict(filter_dict: Optional[Dict[str, object]]) -> Dict[str, List[str]]:
    normalized: Dict[str, List[str]] = {}
    for key, values in (filter_dict or {}).items():
        if values is None:
            continue
        if isinstance(values, (str, int, float)):
            values = [values]
        normalized[str(key)] = [str(value).strip() for value in values]
    return normalized

def build_sample_unit_series(obs: pd.DataFrame, cfg: Dict[str, object]) -> pd.Series:
    sample_unit_key = cfg.get("sample_unit_key")
    if sample_unit_key is not None and str(sample_unit_key) in obs.columns:
        return normalize_string_series(obs[str(sample_unit_key)]).astype(str)

    fallback_keys = [
        str(key)
        for key in cfg.get("sample_unit_fallback_keys", [])
        if str(key) in obs.columns
    ]
    if not fallback_keys:
        fallback_keys = [str(cfg["donor_key"]), str(cfg["analysis_axis_key"])]

    sample_unit = obs[fallback_keys[0]].astype(str)
    for key in fallback_keys[1:]:
        sample_unit = sample_unit + "__" + obs[key].astype(str)
    return sample_unit.astype(str)

def ensure_workflow_config(adata: Any, cfg: Dict[str, object], *, require_latent: bool = True) -> Dict[str, object]:
    cfg = dict(cfg)
    defaults = {
        "disease_axis_key": "L1_Macro",
        "normal_label": "Normal",
        "analysis_axis_key": None,
        "analysis_control_label": None,
        "analysis_case_groups": None,
        "cohort_filters": {},
        "query_key": "query",
        "query_value": "1",
        "source_value": "0",
        "latent_key": "X_scpoliall",
        "broad_group_key": "NK_manno_level2ex",
        "fine_group_key": "NK_manno_level3extend",
        "donor_key": "donor_id",
        "organ_key": "organ",
        "exclude_broad_groups": ["Decidua NK"],
        "exclude_fine_groups": [],
        "disease_groups": ["Tumor", "Infection"],
        "source_k": 10,
        "baseline_min_normal_donors": 5,
        "baseline_min_normal_cells": 100,
        "high_shift_quantile": 0.90,
        "donor_top_quantile": 0.90,
        "state_summary_min_cells_per_unit": 10,
        "sample_unit_key": None,
        "sample_unit_fallback_keys": None,
        "raw_distance_key": "raw_source_distance",
        "atlas_shift_key": "atlas_shift_score",
        "atlas_shift_robust_key": "atlas_shift_score_robust",
        "atlas_high_flag_key": "atlas_shift_high_flag",
        "atlas_exceedance_key": "atlas_exceedance_score",
        "atlas_exceedance_residual_key": "atlas_exceedance_residual",
        "atlas_high_exceedance_flag_key": "atlas_high_exceedance_flag",
    }
    for key, value in defaults.items():
        cfg.setdefault(key, value)

    if cfg["analysis_axis_key"] is None:
        cfg["analysis_axis_key"] = cfg["disease_axis_key"]

    if cfg["analysis_control_label"] is None:
        analysis_axis_key = str(cfg["analysis_axis_key"])
        analysis_values = set(normalize_string_series(adata.obs[analysis_axis_key]).astype(str).unique().tolist())
        if analysis_axis_key == "L4_Standard" and "Healthy (Normal)" in analysis_values:
            cfg["analysis_control_label"] = "Healthy (Normal)"
        else:
            cfg["analysis_control_label"] = cfg["normal_label"]

    if cfg["analysis_case_groups"] is None:
        cfg["analysis_case_groups"] = cfg.get("disease_groups", ["Tumor", "Infection"])

    if cfg["sample_unit_fallback_keys"] is None:
        cfg["sample_unit_fallback_keys"] = [cfg["donor_key"], cfg["analysis_axis_key"]]

    alias_candidates = {
        cfg["fine_group_key"]: ["NK_manno_level3extend", "NK_manno_level3ex2", "NK_manno_level3ex", "nk_level3"],
        cfg["broad_group_key"]: ["NK_manno_level2ex", "NK_manno_level2", "nk_level2"],
        cfg["donor_key"]: ["donor_id", "donor", "sample_id", "sample"],
        cfg["organ_key"]: ["organ", "tissue_general", "Organ_label"],
        cfg["disease_axis_key"]: ["L1_Macro", "disease_level1", "condition"],
        cfg["analysis_axis_key"]: ["L1_Macro", "L4_Standard", "disease_level1", "condition"],
    }
    for preferred_key, aliases in alias_candidates.items():
        if preferred_key in adata.obs.columns:
            continue
        for alias in aliases:
            if alias in adata.obs.columns:
                cfg[preferred_key] = alias
                break

    required_obs = [
        cfg["disease_axis_key"],
        cfg["analysis_axis_key"],
        cfg["query_key"],
        cfg["broad_group_key"],
        cfg["fine_group_key"],
        cfg["donor_key"],
        cfg["organ_key"],
    ]
    required_obs.extend(list(normalize_filter_dict(cfg.get("cohort_filters")).keys()))
    if cfg.get("sample_unit_key"):
        required_obs.append(str(cfg["sample_unit_key"]))
    missing_obs = [key for key in required_obs if key not in adata.obs.columns]
    if missing_obs:
        raise KeyError(f"Missing required obs columns: {missing_obs}")
    if require_latent and cfg["latent_key"] not in adata.obsm:
        raise KeyError(f"Missing latent key {cfg['latent_key']!r} in adata.obsm")

    cfg["disease_groups"] = [str(value) for value in cfg.get("disease_groups", [])]
    cfg["analysis_case_groups"] = [str(value) for value in cfg.get("analysis_case_groups", [])]
    cfg["exclude_broad_groups"] = [str(value) for value in cfg.get("exclude_broad_groups", [])]
    cfg["exclude_fine_groups"] = [str(value) for value in cfg.get("exclude_fine_groups", [])]
    cfg["cohort_filters"] = normalize_filter_dict(cfg.get("cohort_filters"))
    cfg["sample_unit_fallback_keys"] = [str(value) for value in cfg.get("sample_unit_fallback_keys", [])]
    cfg["analysis_sample_unit_key"] = "analysis_sample_unit"
    return cfg

def build_obs_table(adata: Any, cfg: Dict[str, object]) -> pd.DataFrame:
    obs = pd.DataFrame(index=adata.obs_names)
    obs["obs_name"] = obs.index.astype(str)
    obs["obs_position"] = np.arange(adata.n_obs, dtype=int)
    keys_to_copy = {
        str(cfg["query_key"]),
        str(cfg["disease_axis_key"]),
        str(cfg["analysis_axis_key"]),
        str(cfg["broad_group_key"]),
        str(cfg["fine_group_key"]),
        str(cfg["donor_key"]),
        str(cfg["organ_key"]),
        *[str(key) for key in cfg.get("cohort_filters", {}).keys()],
    }
    if cfg.get("sample_unit_key"):
        keys_to_copy.add(str(cfg["sample_unit_key"]))

    for key in sorted(keys_to_copy):
        if key == str(cfg["query_key"]):
            obs[key] = normalize_string_series(adata.obs[key]).astype(str)
        elif key == str(cfg["analysis_axis_key"]):
            obs[key] = normalize_condition_series(adata.obs[key], str(cfg["analysis_control_label"]))
        elif key == str(cfg["disease_axis_key"]):
            obs[key] = normalize_condition_series(adata.obs[key], str(cfg["normal_label"]))
        else:
            obs[key] = normalize_string_series(adata.obs[key]).astype(str)

    obs["analysis_label"] = obs[str(cfg["analysis_axis_key"])].astype(str)
    obs[str(cfg["analysis_sample_unit_key"])] = build_sample_unit_series(obs, cfg)
    return obs

PERTURBATION_CONTROL_DEFAULTS = {'mapping': {'latent_key': 'X_scpoliall',
             'primary_label_key': 'NK_manno_level3extend',
             'secondary_label_key': 'NK_manno_level2ex',
             'condition_keys': ['combined_batch', 'donor_id'],
             'query_label_keys': ['NK_poli_manno_v1',
                                  'NK_poli_manno_low',
                                  'NK_poli_manno_v3'],
             'unknown_label': 'unlabeled',
             'assignment_k': 30,
             'random_seed': 0},
 'design': {'perturbation_key': 'perturbation',
            'donor_id_key': 'donor_id',
            'control_label': 'No stimulation',
            'timepoint_map': {'D2': 'acute', 'D7': 'chronic'},
            'sample_unit_fields': ['donor_id', 'perturbation'],
            'min_cells_per_sample_unit': 20},
 'healthy_reference': {'query_key': 'query', 'query_value': 0, 'k': 30},
 'healthy_timepoint_reference': {'min_control_cells': 20, 'min_control_donors': 2},
 'state_summary': {'label_support_min': 0.6,
             'min_case_cells': 50,
             'min_case_donors': 2,
             'min_cells_per_sample_unit_state': 5,
             'min_case_sample_units': 4,
             'min_state_support_median': 0.75}}


def deep_merge(base, override):
    from copy import deepcopy
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = deep_merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def load_config(path):
    import yaml
    path = Path(path).resolve()
    with path.open() as handle:
        cfg = yaml.safe_load(handle)
    if cfg.get('profile') not in {'matched_control', 'perturbation_control'}:
        raise ValueError('Choose profile explicitly: matched_control or perturbation_control')
    if cfg.get('input_mode') not in {'mapped', 'raw'}:
        raise ValueError('Choose input_mode explicitly: mapped or raw')
    cfg.setdefault('matched_control', {})
    cfg['perturbation_control'] = deep_merge(PERTURBATION_CONTROL_DEFAULTS, cfg.get('perturbation_control', {}))
    cfg['perturbation_control'].setdefault('neighbor_backend', 'pynndescent')
    cfg.setdefault('control_mode', 'internal')
    if cfg['control_mode'] not in {'internal', 'atlas_reference'}:
        raise ValueError('control_mode must be internal or atlas_reference')
    if cfg['profile'] == 'matched_control' and cfg['control_mode'] != 'atlas_reference':
        raise ValueError('matched_control requires control_mode atlas_reference')
    if cfg['control_mode'] == 'atlas_reference' and cfg['profile'] != 'matched_control':
        raise ValueError('atlas_reference requires profile matched_control')
    for key, value in cfg.get('paths', {}).items():
        if value:
            p = Path(value).expanduser()
            cfg['paths'][key] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    return cfg


def read_atlas(path, latent_key='X_scpoliall'):
    import anndata as ad
    import h5py
    from anndata._io.specs import read_elem
    with h5py.File(path, 'r') as f:
        if latent_key not in f['obsm']:
            raise KeyError(f'{path}: missing obsm/{latent_key}')
        a = ad.AnnData(obs=read_elem(f['obs']))
        a.obsm[latent_key] = read_elem(f['obsm'][latent_key])
    if not a.obs_names.is_unique:
        raise ValueError(f'{path}: duplicate cell IDs')
    z = np.asarray(a.obsm[latent_key])
    if z.ndim != 2 or len(z) != a.n_obs or not np.isfinite(z).all():
        raise ValueError(f'{path}: invalid latent matrix')
    return a


def read_cells(path, text_columns=()):
    labels = {'cell_id', 'obs_name', 'donor_id', 'sample_unit', 'analysis_sample_unit', 'L1_Macro', 'L4_Standard', 'organ', 'perturbation', 'timepoint_label', 'NK_manno_level2ex', 'NK_manno_level3extend', *text_columns}
    result = pd.read_csv(path, keep_default_na=False, na_values=[''], float_precision='round_trip', index_col=0, dtype={k: str for k in labels})
    result.index = result.index.astype(str)
    return result


def json_write(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))


def new_output(path):
    p = Path(path)
    if p.exists() and any(p.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {p}; choose a new directory')
    p.mkdir(parents=True, exist_ok=True)
    return p


def check_keys(obs, keys):
    missing = [k for k in keys if k not in obs]
    if missing:
        raise KeyError(f'Missing required metadata: {missing}')
    for key in keys:
        if obs[key].isna().any() or obs[key].astype(str).str.strip().eq('').any():
            raise ValueError(f'Missing values in required metadata: {key}')


def adapt_obs(obs, cfg):
    result = obs.copy()
    for destination, source in cfg.get('obs_columns', {}).items():
        if source not in obs:
            raise KeyError(f'obs_columns source missing: {source}')
        result[destination] = obs[source].copy()
    for column, value in cfg.get('obs_constants', {}).items():
        result[column] = value
    return result
