# Exceedance analysis

## Reference resources

Place the reference bundle in `resources/hnka/`:

```text
resources/hnka/
├── atlas.h5ad
├── model/
│   ├── model_params.pt
│   ├── attr.pkl
│   └── var_names.csv
└── reference/
    ├── reference.json
    ├── geometry.npz
    └── controls.csv.gz
```

The YAML files in `scripts/configs/` specify the input, model, reference and output paths. Relative paths resolve from the configuration file's directory.

## Mapping and analysis

Place raw NK-cell counts in `data/query.h5ad`, with gene symbols in `var_names` and `donor_id` and `combined_batch` in `obs`.

```bash
python scripts/run_exceedance.py map --config scripts/configs/external_query.yaml
python scripts/run_exceedance.py analyze --config scripts/configs/external_query.yaml
```

Analysis saves density and state-shift figures as PNG and PDF in the output directory. To draw figures from existing result tables, use `plot` with the same configuration.

For perturbation analysis, use `scripts/configs/perturbation.yaml` and include `perturbation` and `timepoint_label` in `obs`. This compares perturbations with controls at the same time point. Field names can be adapted with `obs_columns` in the configuration.
