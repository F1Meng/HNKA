from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
if __package__:
    from .engine import inputs as io, scoring as score, summary, mapping
else:
    from engine import inputs as io, scoring as score, summary, mapping


def load_inputs(cfg):
    paths = cfg['paths']; profile = cfg['profile']
    latent_key = cfg['matched_control'].get('latent_key', 'X_scpoliall') if profile == 'matched_control' else cfg['perturbation_control']['mapping']['latent_key']
    atlas = io.read_atlas(paths['atlas_h5ad'], latent_key)
    if cfg['input_mode'] == 'mapped':
        query = io.read_atlas(paths['mapped_h5ad'], latent_key)
        if query.obsm[latent_key].shape[1] != atlas.obsm[latent_key].shape[1]:
            raise ValueError('Atlas/query latent dimensions differ')
    else:
        raise ValueError('Raw input needs the map command first, or analyze will invoke mapping')
    if cfg.get('obs_columns') or cfg.get('obs_constants'):
        query.obs = io.adapt_obs(query.obs, cfg)
    if profile == 'matched_control':
        query.obs[cfg['matched_control'].get('query_key', 'query')] = str(cfg['matched_control'].get('query_value', '1'))
        organ = cfg['matched_control'].get('organ_key', 'organ')
        if organ not in query.obs:
            query.obs[organ] = 'Unknown'
        effective = io.ensure_workflow_config(query, cfg['matched_control'])
        io.check_keys(query.obs, [effective['donor_key'], effective['analysis_axis_key'], effective['broad_group_key'], effective['fine_group_key']])
    else:
        effective = cfg['perturbation_control']
        state = effective['mapping']['primary_label_key']
        io.check_keys(query.obs, [state, state + '_support'])
        support = pd.to_numeric(query.obs[state + '_support'], errors='raise')
        if not np.isfinite(support).all() or not support.between(0, 1).all():
            raise ValueError('Invalid label support')
        query.obs = summary.design_perturbation_control(query.obs, effective)
    return atlas, query, effective


def read_reference(cfg, effective):
    root = Path(cfg['paths']['reference_dir'])
    metadata = json.loads((root / 'reference.json').read_text())
    immutable = ['latent_key', 'source_k', 'query_key', 'source_value', 'broad_group_key', 'fine_group_key', 'organ_key', 'donor_key', 'analysis_axis_key', 'analysis_control_label', 'baseline_min_normal_cells', 'baseline_min_normal_donors', 'exclude_broad_groups', 'exclude_fine_groups', 'cohort_filters', 'high_shift_quantile']
    for key in immutable:
        if effective[key] != metadata['matched_control'][key]:
            raise ValueError(f'Frozen reference configuration mismatch: {key}')
    controls = io.read_cells(root / 'controls.csv.gz', [effective[k] for k in ['donor_key', 'analysis_axis_key', 'broad_group_key', 'fine_group_key', 'organ_key', 'analysis_sample_unit_key']])
    with np.load(root / 'geometry.npz', allow_pickle=False) as f:
        geometry = f['latent'].copy()
    return controls, geometry, metadata


def slug(value):
    import re
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).strip('_') or 'group'


def analyze(cfg):
    outpath = Path(cfg['paths']['output_dir'])
    if outpath.exists() and any(outpath.iterdir()):
        raise FileExistsError(outpath)
    if cfg['input_mode'] == 'raw':
        from copy import deepcopy
        mapped_path = cfg['paths'].get('mapped_h5ad')
        if not mapped_path:
            raise ValueError('Raw analyze needs a new paths.mapped_h5ad output path')
        mapping.map_query(cfg, mapped_path)
        cfg = deepcopy(cfg); cfg['input_mode'] = 'mapped'
    atlas, query, effective = load_inputs(cfg)
    profile = cfg['profile']
    if profile == 'matched_control':
        cells = score.matched_control_scope(query, effective)
        controls, source, _ = read_reference(cfg, effective)
        cells[effective['raw_distance_key']] = score._mean_knn_distance(source, np.asarray(query.obsm[effective['latent_key']], dtype=float)[cells['obs_position'].to_numpy(int)], effective['source_k'])
        cells = score.calibrate_matched_control(cells, effective, controls)
        if controls is not None:
            controls = controls.copy(); controls['reference_role'] = 'atlas_normal'
            cells['reference_role'] = 'query'
            controls['original_cell_id'] = controls.index.astype(str)
            cells['original_cell_id'] = cells.index.astype(str)
            controls.index = '__atlas_normal__' + controls.index.astype(str)
            cells = pd.concat([controls, cells], axis=0)
            if not cells.index.is_unique:
                raise ValueError('Reference/query cell identity collision')
        units = summary.build_sample_score_table(cells, effective)
        tables = summary.build_matched_control_tables(cells, effective)
    else:
        cells = query.obs.copy()
        source, ids = score.source_geometry(atlas, effective, profile)
        geometrical = score.score_perturbation_control_geometry(source, np.asarray(query.obsm[effective['mapping']['latent_key']], dtype=np.float32), effective)
        geometrical.index = cells.index
        for column in geometrical:
            cells[column] = geometrical[column].to_numpy()
        cells = score.assign_healthy_timepoint_exceedance(cells, effective)
        if not cells['healthy_timepoint_exceedance'].notna().any():
            raise ValueError('No eligible timepoint control reference; no Perturbation-control effect can be computed')
        units = summary.perturbation_control_unit_metrics(cells, effective)
        tables = {'state_shifts': summary.standardize(summary.build_state_shifts(cells, effective), profile)}
        if cfg.get('by_perturbation', True):
            by = summary.perturbation_control_by_perturbation(cells, effective)
            if not by.empty:
                tables['by_perturbation'] = summary.standardize(by, profile)
    output = io.new_output(outpath)
    cells.to_csv(output / 'cell_scores.csv.gz', index=True, index_label='cell_id')
    units.to_csv(output / 'sample_metrics.csv', index=False)
    for key, table in tables.items():
        name = '_'.join(slug(v) for v in key) if isinstance(key, tuple) else key
        table.to_csv(output / (name + '.csv'), index=False)
    if profile == 'perturbation_control':
        summary.build_perturbation_exceedance_summary(cells, effective).to_csv(output / 'perturbation_exceedance_summary.csv', index=False)
    if __package__:
        from .engine.plotting import plot_results
    else:
        from engine.plotting import plot_results
    plot_results(cfg)
    return {'output_dir': str(output), 'cells': len(cells)}


def main():
    parser = argparse.ArgumentParser(prog='hnka-exceedance')
    parser.add_argument('command', choices=['map', 'analyze', 'plot'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir')
    parser.add_argument('--reference-dir')
    parser.add_argument('--mapped-h5ad')
    args = parser.parse_args()
    cfg = io.load_config(args.config)
    for value, key in [(args.output_dir, 'output_dir'), (args.reference_dir, 'reference_dir'), (args.mapped_h5ad, 'mapped_h5ad')]:
        if value:
            cfg['paths'][key] = str(Path(value).resolve())
    if args.command == 'map':
        result = {'mapped_h5ad': str(mapping.map_query(cfg, cfg['paths']['mapped_h5ad']))}
    elif args.command == 'plot':
        if __package__:
            from .engine.plotting import plot_results
        else:
            from engine.plotting import plot_results
        result = plot_results(cfg)
    else:
        result = analyze(cfg)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
