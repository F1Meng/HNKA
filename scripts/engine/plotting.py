from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
from urllib.parse import quote

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from . import inputs, summary


def donor_means(cells, cfg):
    if cfg["profile"] == "matched_control":
        settings = inputs.ensure_workflow_config(SimpleNamespace(obs=cells), cfg["matched_control"], require_latent=False)
        group = settings["analysis_axis_key"]
        donor = settings["donor_key"]
        metric = settings["atlas_exceedance_key"]
        keys = [group, donor]
        minimum = int(settings.get("donor_min_cells", 20))
        work = cells
    else:
        settings = cfg["perturbation_control"]
        group = settings["design"]["perturbation_key"]
        donor = settings["design"]["donor_id_key"]
        metric = "healthy_timepoint_exceedance"
        keys = ["timepoint_label", group, donor]
        minimum = int(settings["design"]["min_cells_per_sample_unit"])
        work = summary._distribution_cells(cells, settings)
    work = work.loc[np.isfinite(pd.to_numeric(work[metric], errors="coerce"))]
    means = work.groupby(keys, observed=True)[metric].agg(mean_exceedance="mean", n_cells="size").reset_index()
    means = means.loc[means.n_cells.ge(minimum)]
    return means.rename(columns={group: "group", donor: "donor"})


def _save(fig, prefix):
    try:
        for extension in ["png", "pdf"]:
            fig.savefig(str(prefix) + "." + extension, dpi=300, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)


def atlas_palette(path, state_key):
    import h5py
    from anndata._io.specs import read_elem

    with h5py.File(path, "r") as atlas:
        states = read_elem(atlas["obs"][state_key]).categories.astype(str)
        colors = read_elem(atlas["uns"][state_key + "_colors"])
    if len(states) != len(colors):
        raise ValueError(f"Atlas state/color counts differ for {state_key}")
    return dict(zip(states, colors))


def plot_density(means, prefix, title):
    groups = list(means.groupby("group", observed=True, sort=True))
    fig, ax = plt.subplots(figsize=(8, max(3.2, 1.05 * len(groups) + 1.2)))
    grid = np.linspace(0, 1, 401)
    bandwidth = 0.05
    ticks, labels = [], []
    for i, (group, frame) in enumerate(groups):
        values = frame.mean_exceedance.to_numpy(float)
        base = len(groups) - 1 - i
        color = plt.get_cmap("tab10")(i % 10)
        ticks.append(base + 0.2)
        labels.append(str(group))
        if len(values) >= 2:
            density = np.exp(-0.5 * ((grid[:, None] - values[None, :]) / bandwidth) ** 2).mean(1)
            ridge = density / density.max() * 0.72
            ax.fill_between(grid, base, base + ridge, color=color, alpha=0.35)
            ax.plot(grid, base + ridge, color=color, linewidth=1.2)
            height = max(0.12, float(np.interp(values.mean(), grid, ridge)))
            ax.vlines(values.mean(), base, base + height, color="black", linewidth=1)
        else:
            ax.plot(values, np.full(len(values), base + 0.15), "|", color=color, markersize=12)
            ax.text(0.98, base + 0.2, "Insufficient donors for density", ha="right", fontsize=8)
        ax.hlines(base, 0, 1, color="#cccccc", linewidth=0.5)
    if not groups:
        ax.text(0.5, 0.5, "No eligible donors", transform=ax.transAxes, ha="center")
    ax.set(xlim=(0, 1), yticks=ticks, yticklabels=labels, xlabel="mean exceedance",
           ylabel="Relative density (each peak normalized)", title=title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, prefix)


def plot_shifts(table, state_key, prefix, title, palette, eligible_only=False):
    coordinates = ["mean_exceedance_shift", "high_exceedance_share_shift"]
    valid = np.isfinite(table[coordinates].to_numpy(float)).all(axis=1)
    if eligible_only:
        valid &= table.driver_eligible.fillna(False).to_numpy(bool)
    visible = table.loc[valid].copy()
    fig, (ax, legend_ax) = plt.subplots(
        1, 2, figsize=(10.5, 5.5), constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 0.6], "wspace": 0.04}
    )
    legend_ax.set_axis_off()
    ax.axhline(0, color="#999999", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="#999999", linestyle="--", linewidth=0.8)
    if visible.empty:
        ax.text(0.5, 0.5, "No eligible states with finite shifts", transform=ax.transAxes, ha="center")
    else:
        maximum = max(float(visible.n_case_cells.max()), 1.0)
        sizes = 40 + 340 * np.sqrt(visible.n_case_cells.to_numpy(float) / maximum)
        colors = visible[state_key].astype(str).map(palette)
        ax.scatter(visible[coordinates[0]], visible[coordinates[1]], s=sizes, c=colors,
                   edgecolors="white", linewidths=0.8)
        handles = [Line2D([], [], marker="o", linestyle="", color=palette[state], label=state)
                   for state in sorted(visible[state_key].astype(str).unique())]
        legend = legend_ax.legend(handles=handles, title="NK state", loc="upper left",
                                  frameon=False, fontsize=8)
        legend_ax.add_artist(legend)
        counts = np.unique(np.maximum(1, np.rint(maximum * np.array([0.25, 0.5, 1]))).astype(int))
        sizes = 40 + 340 * np.sqrt(counts / maximum)
        handles = [ax.scatter([], [], s=size, facecolors="none", edgecolors="#777777", label=f"{count:,}")
                   for count, size in zip(counts, sizes)]
        legend_ax.legend(handles=handles, title="Case cells", loc="lower left",
                         frameon=False, fontsize=8, labelspacing=1.2)
        for column, setter in zip(coordinates, [ax.set_xlim, ax.set_ylim]):
            limit = max(float(visible[column].abs().max()), 0.01)
            limit += max(0.015, 0.2 * limit)
            setter(-limit, limit)
    ax.set(xlabel="Mean exceedance shift", ylabel="High-exceedance share shift", title=title)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, prefix)


def plot_results(cfg):
    root = Path(cfg["paths"]["output_dir"])
    destination = root / "figures"
    if destination.exists():
        raise FileExistsError(destination)
    cells = pd.read_csv(root / "cell_scores.csv.gz", index_col="cell_id")
    means = donor_means(cells, cfg)
    with tempfile.TemporaryDirectory(prefix=".plot-", dir=root) as temporary:
        stage = Path(temporary) / "figures"
        stage.mkdir()
        if cfg["profile"] == "matched_control":
            plot_density(means, stage / "mean_exceedance_density", "Reference and query")
            settings = inputs.ensure_workflow_config(SimpleNamespace(obs=cells), cfg["matched_control"], require_latent=False)
            for level, key, palette_key in [("level2", "broad_group_key", "broad_palette"),
                                            ("level3", "fine_group_key", "fine_palette")]:
                state_key = settings[key]
                palette = atlas_palette(cfg["paths"]["atlas_h5ad"], state_key)
                palette.update(settings.get(palette_key, {}))
                for path in sorted(root.glob(level + "_*.csv")):
                    table = pd.read_csv(path)
                    title = str(table.case_group.iloc[0]) if len(table) else level
                    plot_shifts(table, state_key, stage / path.stem, title, palette)
        else:
            table = pd.read_csv(root / "state_shifts.csv")
            state_key = cfg["perturbation_control"]["mapping"]["primary_label_key"]
            palette = atlas_palette(cfg["paths"]["atlas_h5ad"], state_key)
            for timepoint in sorted(cells.timepoint_label.dropna().astype(str).unique()):
                name = quote(timepoint, safe="") or "timepoint"
                plot_density(means.loc[means.timepoint_label.astype(str).eq(timepoint)],
                             stage / ("mean_exceedance_density_" + name), timepoint)
                plot_shifts(table.loc[table.timepoint_label.astype(str).eq(timepoint)], "level3",
                            stage / ("state_shifts_" + name), timepoint, palette, eligible_only=True)
        stage.rename(destination)
    return {"figures_dir": str(destination)}
