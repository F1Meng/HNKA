from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
import pandas as pd
import pickle, random
from .scoring import _build_neighbor_index, _query_neighbor_index

def _clean_string_series(values: Iterable[object]) -> pd.Series:
    series = pd.Series(values, copy=False)
    if isinstance(series.dtype, pd.CategoricalDtype):
        series = series.astype(object)
    return series.where(pd.notna(series), "").astype(str).str.strip()

def _clean_gene_index(values: Iterable[object]) -> pd.Index:
    return pd.Index(_clean_string_series(values).tolist())

def _read_model_gene_index(model_dir: Path) -> pd.Index:
    path = model_dir / "var_names.csv"
    raw = pd.read_csv(path, header=None)
    if raw.shape[1] == 0:
        raise ValueError(f"No model genes found in {path}")
    values = _clean_gene_index(raw.iloc[:, 0])
    values = values[values != ""]
    if values.empty:
        raise ValueError(f"No usable model genes found in {path}")
    return values

def _model_attr(model_dir: Path) -> dict[str, Any]:
    try:
        with (model_dir / "attr.pkl").open("rb") as handle:
            obj = pickle.load(handle)
    except RuntimeError as exc:
        if 'CUDA' in str(exc) or 'cuda' in str(exc):
            raise RuntimeError('This model requires a CUDA device for mapping.') from exc
        raise
    if not isinstance(obj, dict):
        raise TypeError("scPoli attr.pkl must contain a dictionary")
    return obj

def _select_gene_names(query, model_genes):
    candidates = {'var_names': _clean_gene_index(query.var_names)}
    for key in ['feature_name', 'gene_name']:
        if key in query.var:
            candidates[key] = _clean_gene_index(query.var[key])
    model_set = set(model_genes)
    ranked = []
    for name, values in candidates.items():
        usable = values[values != '']
        overlap = len(model_set.intersection(usable))
        ranked.append((-overlap, int(usable.duplicated().sum()), name))
    overlap, _, name = min(ranked)
    if -overlap / len(model_genes) < 0.80:
        raise ValueError('Query/model gene overlap is below 80%')
    return name

def _prepare_query_for_model(query: Any, model_genes: pd.Index, cfg: Mapping[str, Any]) -> Any:
    best_source = _select_gene_names(query, model_genes)

    prepared = query.copy()
    prepared.var["original_var_name"] = prepared.var_names.astype(str)
    if best_source == "var_names":
        prepared.var_names = _clean_gene_index(prepared.var_names)
    else:
        prepared.var_names = _clean_gene_index(prepared.var[best_source])
    prepared.var_names_make_unique()

    for key in cfg["mapping"]["condition_keys"]:
        if key not in prepared.obs.columns:
            raise KeyError(f"Query obs is missing model condition key {key!r}")
        prepared.obs[str(key)] = _clean_string_series(prepared.obs[str(key)]).to_numpy()
    unknown = str(cfg["mapping"].get("unknown_label", "unlabeled"))
    for key in cfg["mapping"]["query_label_keys"]:
        prepared.obs[str(key)] = unknown
    return prepared

def _majority_vote(values: Sequence[object]) -> tuple[str, float]:
    counts = pd.Series(values).value_counts(dropna=False)
    return str(counts.index[0]), float(counts.iloc[0] / counts.sum())

def _assign_atlas_labels(atlas: Any, mapped: Any, cfg: Mapping[str, Any]) -> pd.DataFrame:
    mapping = cfg["mapping"]
    primary = str(mapping["primary_label_key"])
    secondary = str(mapping["secondary_label_key"])
    latent_key = str(mapping["latent_key"])
    assignment_k = int(mapping["assignment_k"])
    seed = int(mapping["random_seed"])

    primary_values = _clean_string_series(atlas.obs[primary])
    atlas_mask = primary_values.ne("").to_numpy()
    atlas_latent = np.asarray(atlas.obsm[latent_key][atlas_mask], dtype=np.float32)
    atlas_primary = primary_values.loc[atlas_mask].to_numpy()
    atlas_secondary = _clean_string_series(atlas.obs.loc[atlas_mask, secondary]).to_numpy()

    max_k = min(assignment_k, atlas_latent.shape[0])
    index, backend = _build_neighbor_index(atlas_latent, max_k, seed, cfg.get('neighbor_backend', 'pynndescent'))
    neighbor_indices, _ = _query_neighbor_index(
        index, np.asarray(mapped.obsm[latent_key]), max_k, backend
    )

    records: list[dict[str, Any]] = []
    for indices in neighbor_indices:
        label_indices = indices[:assignment_k]
        labels = atlas_primary[label_indices]
        counts = pd.Series(labels).value_counts(dropna=False)
        assigned = str(counts.index[0])
        secondary_value, secondary_support = _majority_vote(atlas_secondary[label_indices])
        records.append(
            {
                primary: assigned,
                f"{primary}_support": float(counts.iloc[0] / counts.sum()),
                secondary: secondary_value,
                f"{secondary}_support": secondary_support,
            }
        )
    assignment = pd.DataFrame(records, index=mapped.obs_names)
    overlapping_columns = [column for column in assignment.columns if column in mapped.obs.columns]
    if overlapping_columns:
        mapped.obs = mapped.obs.drop(columns=overlapping_columns)
    mapped.obs = mapped.obs.join(assignment)
    return assignment


def map_query(cfg, output_path):
    import anndata as ad
    import torch
    from scarches.models.scpoli import scPoli
    from .inputs import read_atlas, adapt_obs
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(path)
    settings = cfg['perturbation_control']
    seed = int(settings['mapping']['random_seed'])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    atlas = read_atlas(cfg['paths']['atlas_h5ad'], settings['mapping']['latent_key'])
    query = ad.read_h5ad(cfg['paths']['query_h5ad'])
    query.obs = adapt_obs(query.obs, cfg)
    if not query.obs_names.is_unique:
        raise ValueError('Query cell IDs must be unique')
    model_dir = Path(cfg['paths']['model_dir'])
    genes = _read_model_gene_index(model_dir)
    attr = _model_attr(model_dir)
    for attrkey, cfgkey in [('condition_keys_', 'condition_keys'), ('cell_type_keys_', 'query_label_keys')]:
        if list(attr.get(attrkey, [])) != list(settings['mapping'][cfgkey]):
            raise ValueError(f'Model {attrkey} does not match mapping configuration')
    prepared = _prepare_query_for_model(query, genes, settings)
    model = scPoli.load_query_data(adata=prepared, reference_model=str(model_dir), labeled_indices=[], unknown_ct_names=[settings['mapping']['unknown_label']], map_location=None if torch.cuda.is_available() else torch.device('cpu'))
    model.model.eval()
    if torch.cuda.is_available():
        model.model.cuda(); model.model.eval()
    if not query.obs_names.astype(str).equals(model.adata.obs_names.astype(str)):
        raise ValueError('Mapping changed query cell order')
    if not _clean_gene_index(model.adata.var_names).equals(genes):
        raise ValueError('Model gene order differs from validated query')
    latent = np.asarray(model.get_latent(model.adata, mean=True), dtype=np.float32)
    if latent.shape != (query.n_obs, int(attr.get('latent_dim_', latent.shape[1]))):
        raise ValueError('Mapped latent dimensions differ from model')
    if not np.isfinite(latent).all():
        raise ValueError('Nonfinite mapped latent values')
    query.obsm[settings['mapping']['latent_key']] = latent
    _assign_atlas_labels(atlas, query, settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    query.write_h5ad(path)
    return path
