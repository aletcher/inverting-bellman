"""Aggregate (depth × width × seed) sweep results into joint plots.

Reads each cell's pqn_checkpoint.pkl (metrics["eval_returns_per_goal"]) and
wm_track/tracking.npz (per-checkpoint WM dynamics MSE), groups by (D, W), and
produces:

  * pqn_return_by_width_d{D*}.png   — fix D=D*, one curve per width
  * pqn_return_by_depth_w{W*}.png   — fix W=W*, one curve per depth
  * wm_dyn_mse_by_width_d{D*}.png   — fix D=D*, one curve per width (log y)
  * wm_dyn_mse_by_depth_w{W*}.png   — fix W=W*, one curve per depth (log y)
  * pqn_return_full_grid.png        — heatmap (D, W) of final-step return
  * wm_dyn_mse_full_grid.png        — heatmap (D, W) of final WM dyn_mse
  * per_run_metrics.npz             — all curves stacked, (D, W)-keyed

Slice plots show mean ± SE across seeds. Style via plotting/style.py
("reacher architecture sweep"). --paper_mode for camera-ready output;
--curve_palette tab10 for the qualitative seaborn palette.
"""

import glob
import os
import pickle
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from plotting.style import (
    ARCH_CMAP_HEATMAP, ARCH_CMAP_PQN_RETURN, ARCH_CMAP_WM_MSE,
    ARCH_CMAP_LO, ARCH_CMAP_HI, ARCH_CURVE_MARKER_KW,
    PAPER_TEXTWIDTH, PAPER_FIG_HEIGHT, PALETTE,
    sample_palette, set_paper_style, unset_paper_style,
)


def plot_mean_se(ax, x, data, color, label, marker_kw=None, **kw):
    """Plot mean line with shaded SE band. data: (num_seeds, N) or (N,)."""
    plot_kw = dict(marker_kw or {}, **kw)
    data = np.array(data)
    if data.ndim == 2 and data.shape[0] > 1:
        mean = data.mean(axis=0)
        se = data.std(axis=0) / np.sqrt(data.shape[0])
        ax.plot(x, mean, color=color, label=label, **plot_kw)
        ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.2)
    else:
        if data.ndim == 2:
            data = data[0]
        ax.plot(x, data, color=color, label=label, **plot_kw)


CELL_RE = re.compile(r"d(\d+)_w(\d+)_s(\d+)")


def aggregate_arch_sweep(
    sweep_dir,
    pivot_depth=4,
    pivot_width=512,
    save_dir=None,
    metric="final_dyn_mse",
    paper_mode=False,
    curve_palette="sequential",  # "sequential" | "tab10"
    cmap_heatmap=None,           # any matplotlib cmap; None → ARCH_CMAP_HEATMAP
    wm_subdir="wm_track",
    depths_filter=None,          # only include cells with D in this set
    widths_filter=None,          # only include cells with W in this set
    unseen_return_kind="disc",   # "disc" or "undisc" — selects which return field
                                 # the unseen-goal heatmaps + headlines are built from
    show_cbar=True,              # if False, drop the colorbar entirely from heatmaps
    log_pqn=False,               # if True, plot PQN return slice + heatmap on log scale
    wm_mse_ymax=None,            # if set, cap WM-MSE slice y-axis + heatmap colorscale at this value
    unseen_only_labels=None,     # if set, restrict unseen-goal aggregation to this iterable of labels
):
    """Walk sweep_dir, build per-(D, W) curves, write 4 slice plots + 2 heatmaps.

    Args:
        sweep_dir: root containing d{D}_w{W}_s{S}/ cell subdirectories.
        pivot_depth, pivot_width: fix these axes for the slice plots.
        save_dir: default depends on wm_subdir — "aggregate" for canonical
            "wm_track", "aggregate_<suffix>" otherwise (so variants don't
            clobber each other's plots / npz).
        metric: WM tracking field to plot.
        paper_mode: if True, drop titles and use paper fig dimensions.
        curve_palette: "sequential" (Blues for PQN, Oranges for WM) or "tab10"
            (seaborn qualitative).
        cmap_heatmap: heatmap colormap. None → ARCH_CMAP_HEATMAP.
        wm_subdir: per-cell subdir to load WM tracking from (default "wm_track";
            "wm_track_policy" for the policy-sampled variant).
        depths_filter, widths_filter: iterables of ints; if provided, only cells
            with D / W in these sets are aggregated. Pair with save_dir to keep
            filtered output separate from the full grid.
    """
    if save_dir is None:
        if wm_subdir == "wm_track":
            save_dir = os.path.join(sweep_dir, "aggregate")
        else:
            # "wm_track_policy" → "aggregate_policy"; else full subdir name.
            suffix = wm_subdir[len("wm_track_"):] if wm_subdir.startswith("wm_track_") else wm_subdir
            save_dir = os.path.join(sweep_dir, f"aggregate_{suffix}")
    os.makedirs(save_dir, exist_ok=True)

    cells = _discover_cells(sweep_dir)
    if not cells:
        print(f"No cells found in {sweep_dir}/d*_w*_s*/")
        return
    if depths_filter is not None:
        depths_filter = set(int(d) for d in depths_filter)
        cells = [(D, W, S) for (D, W, S) in cells if D in depths_filter]
    if widths_filter is not None:
        widths_filter = set(int(w) for w in widths_filter)
        cells = [(D, W, S) for (D, W, S) in cells if W in widths_filter]
    if not cells:
        print(f"No cells survived filter (depths={depths_filter}, "
              f"widths={widths_filter}); aborting.")
        return
    print(f"Found {len(cells)} cells across "
          f"{len(set((D, W) for D, W, _ in cells))} (D, W) configs "
          f"(wm_subdir={wm_subdir!r}, save_dir={save_dir!r})")
    if depths_filter is not None or widths_filter is not None:
        print(f"  Filter: depths={sorted(depths_filter) if depths_filter else 'all'}, "
              f"widths={sorted(widths_filter) if widths_filter else 'all'}")

    # Load per-cell curves.
    pqn_by_dw, pqn_steps_by_dw = _load_pqn_curves(sweep_dir, cells)
    wm_by_dw, wm_steps_by_dw = _load_wm_curves(
        sweep_dir, cells, metric=metric, subdir=wm_subdir,
    )

    # Save consolidated npz.
    consolidated = {}
    for (D, W), curves in pqn_by_dw.items():
        consolidated[f"pqn_return_d{D}_w{W}"] = curves
        consolidated[f"pqn_steps_d{D}_w{W}"] = pqn_steps_by_dw[(D, W)]
    for (D, W), curves in wm_by_dw.items():
        consolidated[f"wm_dyn_d{D}_w{W}"] = curves
        consolidated[f"wm_steps_d{D}_w{W}"] = wm_steps_by_dw[(D, W)]
    npz_path = os.path.join(save_dir, "per_run_metrics.npz")
    np.savez(npz_path, **consolidated)
    print(f"Saved {npz_path}")

    heatmap_cmap = cmap_heatmap if cmap_heatmap is not None else ARCH_CMAP_HEATMAP

    set_paper_style()
    try:
        # Slice plots.
        _plot_slice(
            curves_by_dw=pqn_by_dw, steps_by_dw=pqn_steps_by_dw,
            fix_axis="D", fix_value=pivot_depth,
            ylabel="Discounted return",
            title=f"PQN return by width (D={pivot_depth})",
            save_path=os.path.join(save_dir, f"pqn_return_by_width_d{pivot_depth}.png"),
            legend_prefix="W=", log_y=log_pqn,
            cmap=ARCH_CMAP_PQN_RETURN,
            palette_mode=curve_palette, paper_mode=paper_mode,
        )
        _plot_slice(
            curves_by_dw=pqn_by_dw, steps_by_dw=pqn_steps_by_dw,
            fix_axis="W", fix_value=pivot_width,
            ylabel="Discounted return",
            title=f"PQN return by depth (W={pivot_width})",
            save_path=os.path.join(save_dir, f"pqn_return_by_depth_w{pivot_width}.png"),
            legend_prefix="D=", log_y=log_pqn,
            cmap=ARCH_CMAP_PQN_RETURN,
            palette_mode=curve_palette, paper_mode=paper_mode,
        )
        _plot_slice(
            curves_by_dw=wm_by_dw, steps_by_dw=wm_steps_by_dw,
            fix_axis="D", fix_value=pivot_depth,
            ylabel="World Model MSE",
            title=f"World Model MSE by width (D={pivot_depth})",
            save_path=os.path.join(save_dir, f"wm_dyn_mse_by_width_d{pivot_depth}.png"),
            legend_prefix="W=", log_y=True,
            cmap=ARCH_CMAP_WM_MSE,
            palette_mode=curve_palette, paper_mode=paper_mode,
            ymax=wm_mse_ymax,
        )
        _plot_slice(
            curves_by_dw=wm_by_dw, steps_by_dw=wm_steps_by_dw,
            fix_axis="W", fix_value=pivot_width,
            ylabel="World Model MSE",
            title=f"World Model MSE by depth (W={pivot_width})",
            save_path=os.path.join(save_dir, f"wm_dyn_mse_by_depth_w{pivot_width}.png"),
            legend_prefix="D=", log_y=True,
            cmap=ARCH_CMAP_WM_MSE,
            palette_mode=curve_palette, paper_mode=paper_mode,
            ymax=wm_mse_ymax,
        )

        # Heatmaps over the full (D, W) grid. Seeds-count read from data so the
        # label tracks the sweep.
        n_seeds = next(iter(pqn_by_dw.values())).shape[0]
        _plot_heatmap(
            pqn_by_dw, agg="last",
            cbar_label=f"Discounted return ({n_seeds} seeds)",
            title="PQN return at final step",
            save_path=os.path.join(save_dir, "pqn_return_full_grid.png"),
            log_color=log_pqn, cmap=heatmap_cmap, paper_mode=paper_mode,
            show_cbar=show_cbar,
        )
        _plot_heatmap(
            wm_by_dw, agg="last",
            cbar_label=f"World Model MSE ({n_seeds} seeds)",
            title="World Model MSE at final PQN checkpoint",
            save_path=os.path.join(save_dir, "wm_dyn_mse_full_grid.png"),
            log_color=True, cmap=heatmap_cmap, paper_mode=paper_mode,
            show_cbar=show_cbar, vmax=wm_mse_ymax,
        )
    finally:
        unset_paper_style()

    # Unseen-goal aggregation. Auto-discovers any unseen_goals_arch* subdirs so
    # multiple sampler variants get separate plot dirs.
    unseen_subdirs = _discover_unseen_subdirs(sweep_dir, cells)
    for usub in unseen_subdirs:
        # 'unseen_goals_arch' → 'aggregate/unseen';
        # 'unseen_goals_arch_reset' → 'aggregate/unseen_reset'.
        if usub == "unseen_goals_arch":
            plot_subdir = "unseen"
        elif usub.startswith("unseen_goals_arch_"):
            plot_subdir = "unseen_" + usub[len("unseen_goals_arch_"):]
        else:
            plot_subdir = usub  # fallback
        usub_save_dir = os.path.join(save_dir, plot_subdir) if save_dir else None
        headline_unseen_dw = aggregate_unseen(
            sweep_dir,
            save_dir=usub_save_dir,
            cells=cells,
            cmap_heatmap=heatmap_cmap,
            paper_mode=paper_mode,
            unseen_subdir=usub,
            return_kind=unseen_return_kind,
            show_cbar=show_cbar,
            only_labels=unseen_only_labels,
        )

        # Combined three-panel paper plot: PQN return | WM dyn MSE | unseen return.
        # One per sampler variant, next to its unseen-subdir plots.
        if headline_unseen_dw is not None and usub_save_dir is not None:
            kind_lbl = "disc." if unseen_return_kind == "disc" else "undisc."
            set_paper_style()
            try:
                three_panel_path = os.path.join(usub_save_dir, "combined_three_panel.png")
                _plot_three_panel_heatmap(
                    panels=[
                        dict(curves=pqn_by_dw, log_color=log_pqn,
                             cbar_label="Discounted return",
                             title="PQN Return"),
                        dict(curves=wm_by_dw, log_color=True,
                             cbar_label="World Model MSE",
                             title="World Model MSE Error"),
                        dict(curves={k: v[:, None] for k, v in headline_unseen_dw.items()},
                             log_color=False,
                             cbar_label=f"Unseen-goal {kind_lbl} return",
                             title="WM-Policy Return on Unseen Goals"),
                    ],
                    save_path=three_panel_path,
                    cmap=heatmap_cmap,
                    paper_mode=paper_mode,
                )
                # Mirror the headline paper PNG into outputs/figures/ alongside the
                # other figure scripts' outputs. The full aggregate/ tree (slice
                # plots, per-run npz, ...) stays in-place.
                import shutil
                figures_dir = "outputs/figures"
                os.makedirs(figures_dir, exist_ok=True)
                shutil.copy(
                    three_panel_path,
                    os.path.join(figures_dir, "fig4_arch_sweep_combined_three_panel.png"),
                )
            finally:
                unset_paper_style()


def aggregate_unseen(
    sweep_dir,
    save_dir=None,
    cells=None,
    cmap_heatmap=None,
    paper_mode=False,
    unseen_subdir="unseen_goals_arch",
    return_kind="disc",
    show_cbar=True,
    only_labels=None,
):
    """Aggregate per-cell unseen-goal results into per-goal plots + headline.

    Reads {sweep_dir}/d*_w*_s*/{unseen_subdir}/unseen_summary.npz. For each
    goal label, builds {(D, W): (n_seeds,)} arrays of WM-policy and optimal
    return, then produces:
      * <label>_return_wm_full_grid.png        heatmap of WM-policy return.
      * headline_unseen_return_full_grid.png   mean WM-policy return across goals.

    unseen_subdir: per-cell dir holding the unseen results. 'unseen_goals_arch'
    default; 'unseen_goals_arch_reset' / '_policy' for sampler-tagged variants.

    return_kind: "disc" (default) reads return_*_disc_mean; "undisc" reads
    return_*_undisc_mean. Plot-time selector only — both are saved per-cell.

    only_labels: if set, restrict aggregation to this iterable of labels (e.g.
    ["pure_velocity", "target_angle", "offgrid_fingertip"]) for both per-goal
    plots and the headline mean.
    """
    if return_kind not in ("disc", "undisc"):
        raise ValueError(f"return_kind must be 'disc' or 'undisc'; got {return_kind!r}")
    if cells is None:
        cells = _discover_cells(sweep_dir)
    save_dir = save_dir or os.path.join(sweep_dir, "aggregate", "unseen")

    # Load per-cell unseen summaries.
    by_label_wm = defaultdict(dict)        # {label: {(D, W): np.array shape (n_seeds,)}}
    by_label_true = defaultdict(dict)
    seeds_buf_wm = defaultdict(lambda: defaultdict(list))   # label → (D, W) → list[float]
    seeds_buf_true = defaultdict(lambda: defaultdict(list))
    wm_field = f"return_wm_{return_kind}_mean"
    true_field = f"return_true_{return_kind}_mean"
    found_any = False
    for (D, W, S) in cells:
        path = os.path.join(sweep_dir, f"d{D}_w{W}_s{S}",
                            unseen_subdir, "unseen_summary.npz")
        if not os.path.exists(path):
            continue
        found_any = True
        d = np.load(path, allow_pickle=False)
        labels = list(d["labels"])
        for i, label in enumerate(labels):
            label = str(label)
            if only_labels is not None and label not in only_labels:
                continue
            seeds_buf_wm[label][(D, W)].append(float(d[wm_field][i]))
            seeds_buf_true[label][(D, W)].append(float(d[true_field][i]))

    if not found_any:
        print(f"No unseen_summary.npz found under {sweep_dir}/d*_w*_s*/ "
              f"— skipping unseen aggregation.")
        return

    os.makedirs(save_dir, exist_ok=True)
    print(f"\n── Unseen-goal aggregation ({len(seeds_buf_wm)} goal labels) ──")

    # Stack lists → arrays.
    for label in seeds_buf_wm:
        for (D, W), vals in seeds_buf_wm[label].items():
            by_label_wm[label][(D, W)] = np.array(vals)
        for (D, W), vals in seeds_buf_true[label].items():
            by_label_true[label][(D, W)] = np.array(vals)

    set_paper_style()
    try:
        heatmap_cmap = cmap_heatmap if cmap_heatmap is not None else ARCH_CMAP_HEATMAP
        kind_lbl = "disc." if return_kind == "disc" else "undisc."

        # _plot_heatmap expects (n_seeds, K) and uses [:, -1] per seed. Unseen
        # data is already collapsed to a single value per seed, so we just add
        # a length-1 trailing axis.
        def _to_curves(d):
            return {k: v[:, None] for k, v in d.items()}

        # Per-goal heatmaps + collect per-label arrays for the headlines.
        per_goal_summary = {}
        for label in sorted(seeds_buf_wm.keys()):
            wm_dw  = by_label_wm[label]    # {(D, W): (n_seeds,)}
            true_dw = by_label_true[label]

            _plot_heatmap(
                _to_curves(wm_dw), agg="last",
                cbar_label=f"WM-policy {kind_lbl} return (mean over seeds)",
                title=f"WM-policy {kind_lbl} return - {label}",
                save_path=os.path.join(save_dir, f"{label}_return_wm_full_grid.png"),
                log_color=False, cmap=heatmap_cmap, paper_mode=paper_mode,
                show_cbar=show_cbar,
            )
            per_goal_summary[label] = {
                "wm_return_mean":   {k: float(v.mean()) for k, v in wm_dw.items()},
                "true_return_mean": {k: float(v.mean()) for k, v in true_dw.items()},
            }

        # Headline aggregates: stack per-goal (n_seeds,) into (n_goals, n_seeds)
        # per cell, mean over goals (keep seed dim), then plot mean ± SE.
        # If some goals at a cell have fewer seeds (e.g. added mid-sweep), drop
        # the short ones at that cell rather than truncating all goals to the
        # min — otherwise the headline collapses to n=1 even though per-goal
        # heatmaps show n=3+.
        _dropped_log = []  # (label, (D,W), kept_n, this_n) — printed once after
        def _avg_over_goals(by_label_dict):
            keys = set()
            for label in by_label_dict:
                keys |= set(by_label_dict[label].keys())
            out = {}
            for k in keys:
                stacks = []
                lbls = []
                for label in by_label_dict:
                    if k not in by_label_dict[label]:
                        continue
                    stacks.append(by_label_dict[label][k])
                    lbls.append(label)
                if not stacks:
                    continue
                max_n = max(s.size for s in stacks)
                full, full_lbls = [], []
                for s, lbl in zip(stacks, lbls):
                    if s.size == max_n:
                        full.append(s)
                        full_lbls.append(lbl)
                    else:
                        _dropped_log.append((lbl, k, max_n, s.size))
                out[k] = np.stack(full).mean(axis=0)
            return out

        headline_return = _avg_over_goals(by_label_wm)

        # Surface (label, cell) pairs dropped from the headline so the user
        # knows the mean isn't covering every goal at every cell.
        if _dropped_log:
            from collections import Counter
            ctr = Counter((lbl, max_n, this_n) for (lbl, _, max_n, this_n) in _dropped_log)
            print("  [headline] dropped sparse (label, cell) entries from "
                  "the across-goals mean (label has fewer seeds than the cell's max):")
            for (lbl, max_n, this_n), count in sorted(ctr.items()):
                print(f"    {lbl}: {count} cell(s) where this label has n={this_n} "
                      f"vs other labels' n={max_n}")

        _plot_heatmap(
            _to_curves(headline_return), agg="last",
            cbar_label=f"Mean WM-policy {kind_lbl} return  (averaged over unseen goals)",
            title=f"WM-policy {kind_lbl} return on unseen goals",
            save_path=os.path.join(save_dir, "headline_unseen_return_full_grid.png"),
            log_color=False, cmap=heatmap_cmap, paper_mode=paper_mode,
            show_cbar=show_cbar,
        )
    finally:
        unset_paper_style()

    # Persist consolidated npz for downstream use.
    flat = {}
    for label, blocks in per_goal_summary.items():
        for kind, kv in blocks.items():
            for (D, W), v in kv.items():
                flat[f"{label}/{kind}/d{D}_w{W}"] = np.array(v)
    np.savez(os.path.join(save_dir, "per_run_unseen.npz"), **flat)
    print(f"  Saved unseen aggregate to {save_dir}/")
    # Return the headline-return dict so callers (e.g. the combined paper plot)
    # can reuse the (D, W) → (n_seeds,) array without re-loading.
    return headline_return


def _discover_cells(sweep_dir):
    cells = []
    for entry in sorted(os.listdir(sweep_dir)):
        m = CELL_RE.match(entry)
        if not m:
            continue
        D, W, S = int(m.group(1)), int(m.group(2)), int(m.group(3))
        cells.append((D, W, S))
    return cells


def _discover_unseen_subdirs(sweep_dir, cells):
    """Sorted list of distinct 'unseen_goals_arch*' subdirs present across cells."""
    found = set()
    for (D, W, S) in cells:
        cell_path = os.path.join(sweep_dir, f"d{D}_w{W}_s{S}")
        if not os.path.isdir(cell_path):
            continue
        for entry in os.listdir(cell_path):
            if not entry.startswith("unseen_goals_arch"):
                continue
            full = os.path.join(cell_path, entry, "unseen_summary.npz")
            if os.path.exists(full):
                found.add(entry)
    return sorted(found)


def _load_pqn_curves(sweep_dir, cells):
    """Return ({(D, W): (n_seeds, K)}, {(D, W): (K,) eval steps})."""
    seed_buf = defaultdict(list)
    steps_ref = {}
    for (D, W, S) in cells:
        path = os.path.join(sweep_dir, f"d{D}_w{W}_s{S}", "pqn_checkpoint.pkl")
        if not os.path.exists(path):
            print(f"  WARN: missing {path}")
            continue
        with open(path, "rb") as f:
            data = pickle.load(f)
        metrics = data.get("metrics") or {}
        if "eval_returns_per_goal" not in metrics:
            print(f"  WARN: no eval_returns_per_goal in {path}")
            continue
        eval_arr = np.asarray(metrics["eval_returns_per_goal"])  # (T, num_goals)
        # Keep rows where eval ran (any goal non-NaN).
        eval_mask = ~np.isnan(eval_arr[:, 0])
        steps = np.where(eval_mask)[0].astype(np.int64)
        eval_data = eval_arr[eval_mask]                           # (K, num_goals)
        mean_over_goals = np.nanmean(eval_data, axis=1)           # (K,)
        seed_buf[(D, W)].append(mean_over_goals)
        steps_ref.setdefault((D, W), steps)
    # Stack per (D, W). Truncate to the shortest curve (partial runs).
    out = {}
    out_steps = {}
    for (D, W), curves in seed_buf.items():
        K = min(c.shape[0] for c in curves)
        stacked = np.stack([c[:K] for c in curves])  # (n_seeds, K)
        out[(D, W)] = stacked
        out_steps[(D, W)] = steps_ref[(D, W)][:K]
    return out, out_steps


def _load_wm_curves(sweep_dir, cells, metric="final_dyn_mse", subdir="wm_track"):
    seed_buf = defaultdict(list)
    steps_ref = {}
    for (D, W, S) in cells:
        path = os.path.join(sweep_dir, f"d{D}_w{W}_s{S}", subdir, "tracking.npz")
        if not os.path.exists(path):
            continue
        d = np.load(path)
        if metric not in d:
            print(f"  WARN: metric {metric!r} not in {path}")
            continue
        seed_buf[(D, W)].append(np.asarray(d[metric]))
        steps_ref.setdefault((D, W), np.asarray(d["n_updates"]))
    out = {}
    out_steps = {}
    for (D, W), curves in seed_buf.items():
        K = min(c.shape[0] for c in curves)
        stacked = np.stack([c[:K] for c in curves])
        out[(D, W)] = stacked
        out_steps[(D, W)] = steps_ref[(D, W)][:K]
    return out, out_steps


def _curve_colors(keys, cmap, palette_mode):
    """One color per curve, light→dark in keys order. "sequential" samples cmap
    (Blues/Oranges, lo=0.4, hi=0.9); "tab10" returns the first n PALETTE entries."""
    n = len(keys)
    if palette_mode == "tab10":
        return list(PALETTE[:n])
    if palette_mode == "sequential":
        return sample_palette(cmap, n, lo=ARCH_CMAP_LO, hi=ARCH_CMAP_HI)
    raise ValueError(f"Unknown palette_mode={palette_mode!r}")


def _plot_slice(
    curves_by_dw, steps_by_dw, fix_axis, fix_value,
    ylabel, title, save_path, legend_prefix, log_y,
    cmap, palette_mode, paper_mode,
    ymax=None,
):
    """Plot one curve per varying-axis value at the given pivot.

    ymax: cap the y-axis upper limit. Useful for log-scale WM-MSE plots where
    bad-fit cells (MSE > 1e-2) compress the visible range; set ymax=1e-2 to
    clip them and zoom on the meaningful tail.
    """
    if fix_axis == "D":
        keys = sorted(W for (D, W) in curves_by_dw if D == fix_value)
        get = lambda v: (fix_value, v)
    elif fix_axis == "W":
        keys = sorted(D for (D, W) in curves_by_dw if W == fix_value)
        get = lambda v: (v, fix_value)
    else:
        raise ValueError(fix_axis)

    if not keys:
        print(f"  No data for {fix_axis}={fix_value}; skipping {save_path}")
        return

    colors = _curve_colors(keys, cmap, palette_mode)

    fig, ax = plt.subplots(figsize=(PAPER_TEXTWIDTH, PAPER_FIG_HEIGHT))

    # Ascending so the largest condition (darkest sequential) draws last → on top.
    for i, v in enumerate(keys):
        D, W = get(v)
        curves = curves_by_dw[(D, W)]
        steps = steps_by_dw[(D, W)]
        if curves.shape[0] == 0:
            continue
        plot_mean_se(
            ax, steps, curves, colors[i], f"{legend_prefix}{v}",
            **(ARCH_CURVE_MARKER_KW or {}),
        )

    ax.set_xlabel("PQN update step")
    ax.set_ylabel(ylabel)
    if not paper_mode:
        ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    if ymax is not None:
        ax.set_ylim(top=ymax)
    ax.legend(framealpha=0.9, loc="best")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")


def _plot_heatmap(
    curves_by_dw, agg, cbar_label, title, save_path, log_color,
    cmap, paper_mode, show_cbar=True, vmax=None,
):
    """Heatmap over (D, W) grid of an aggregated final-step value.

    Cell text shows mean (top) and ±SE (below, if n > 1). Text color flips
    between white and black based on the cell's normalized luminance.

    vmax: cap the colorscale upper bound. Useful for log-color WM-MSE heatmaps
    where bad cells (MSE > 1e-2) saturate the palette and flatten the gradient
    over the meaningful tail.
    """
    Ds = sorted({D for (D, W) in curves_by_dw})
    Ws = sorted({W for (D, W) in curves_by_dw})
    if not Ds or not Ws:
        print(f"  No grid data for {save_path}; skipping")
        return

    Z = np.full((len(Ds), len(Ws)), np.nan)
    SE = np.full((len(Ds), len(Ws)), np.nan)
    N = np.zeros((len(Ds), len(Ws)), dtype=int)
    for i, D in enumerate(Ds):
        for j, W in enumerate(Ws):
            curves = curves_by_dw.get((D, W))
            if curves is None or curves.shape[0] == 0:
                continue
            if agg == "last":
                last = curves[:, -1]
            else:
                raise ValueError(agg)
            n = last.shape[0]
            N[i, j] = n
            Z[i, j] = float(np.nanmean(last))
            if n > 1:
                SE[i, j] = float(np.nanstd(last, ddof=1) / np.sqrt(n))

    # Locally swap to sns "white" style — black spines, no grid lines through
    # cells. set_paper_style's font / dpi rcParams stay in effect; slice plots
    # keep whitegrid.
    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(PAPER_TEXTWIDTH, PAPER_FIG_HEIGHT))
        if log_color:
            from matplotlib.colors import LogNorm
            vmin_eff = float(np.nanmin(Z[Z > 0])) if np.any(Z > 0) else 1e-12
            vmax_eff = vmax if vmax is not None else float(np.nanmax(Z))
            norm = LogNorm(vmin=vmin_eff, vmax=vmax_eff)
        else:
            from matplotlib.colors import Normalize
            norm = Normalize(vmin=float(np.nanmin(Z)), vmax=float(np.nanmax(Z)))
        im = ax.imshow(Z, aspect="auto", origin="lower", cmap=cmap, norm=norm)

        ax.set_xticks(range(len(Ws)))
        ax.set_xticklabels([str(w) for w in Ws])
        ax.set_yticks(range(len(Ds)))
        ax.set_yticklabels([str(d) for d in Ds])
        ax.set_xlabel("Network Width")
        ax.set_ylabel("Network Depth")
        if not paper_mode:
            ax.set_title(title)

        for i in range(len(Ds)):
            for j in range(len(Ws)):
                v = Z[i, j]
                if np.isnan(v):
                    continue
                se = SE[i, j]
                # mathtext \pm so the symbol renders even when the active font
                # (e.g. cmr10) lacks "±". Scientific notation for log-scale
                # (orders of magnitude); decimal for linear (O(1)).
                if log_color:
                    mean_str, se_str = f"{v:.1e}", f"{se:.0e}"
                else:
                    mean_str, se_str = f"{v:.2f}", f"{se:.1g}"
                txt = mean_str if np.isnan(se) else f"{mean_str}\n$\\pm${se_str}"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white", fontsize=10 if paper_mode else 8)

        if show_cbar:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(cbar_label)
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
    print(f"Saved {save_path}")


def _plot_three_panel_heatmap(panels, save_path, cmap, paper_mode):
    """Render N heatmaps side-by-side for a paper-paper figure.

    Each panel is a dict {curves, log_color, cbar_label, title}; curves maps
    (D, W) → (n_seeds, K) and is reduced to per-cell mean / SE on the final K.
    Network Depth label + ticks only on the leftmost panel; every panel has its
    own colorbar (ticks but no label — label lives in the figure caption).
    """
    if not panels:
        return
    n = len(panels)

    with sns.axes_style("white"):
        # Each subplot needs ~PAPER_TEXTWIDTH width so the per-cell mean ± SE
        # text doesn't collide. Figure scales linearly with n.
        fig, axes = plt.subplots(
            1, n,
            figsize=(PAPER_TEXTWIDTH * 0.95 * n, PAPER_FIG_HEIGHT * 1.0),
            constrained_layout=True,
        )
        # Tighten wspace so panels sit close (cbar tick labels still fit).
        fig.set_constrained_layout_pads(w_pad=0.0, wspace=0.02)
        if n == 1:
            axes = [axes]

        for i, p in enumerate(panels):
            ax = axes[i]
            curves_by_dw = p["curves"]
            log_color = p.get("log_color", False)

            Ds = sorted({D for (D, W) in curves_by_dw})
            Ws = sorted({W for (D, W) in curves_by_dw})
            if not Ds or not Ws:
                ax.set_axis_off()
                continue

            Z = np.full((len(Ds), len(Ws)), np.nan)
            SE = np.full((len(Ds), len(Ws)), np.nan)
            for ii, D in enumerate(Ds):
                for jj, W in enumerate(Ws):
                    cu = curves_by_dw.get((D, W))
                    if cu is None or cu.shape[0] == 0:
                        continue
                    last = cu[:, -1]
                    Z[ii, jj] = float(np.nanmean(last))
                    if last.shape[0] > 1:
                        SE[ii, jj] = float(
                            np.nanstd(last, ddof=1) / np.sqrt(last.shape[0])
                        )

            if log_color:
                from matplotlib.colors import LogNorm
                vmin = float(np.nanmin(Z[Z > 0])) if np.any(Z > 0) else 1e-12
                vmax = float(np.nanmax(Z))
                norm = LogNorm(vmin=vmin, vmax=vmax)
            else:
                from matplotlib.colors import Normalize
                norm = Normalize(vmin=float(np.nanmin(Z)),
                                 vmax=float(np.nanmax(Z)))
            im = ax.imshow(Z, aspect="auto", origin="lower",
                           cmap=cmap, norm=norm)

            ax.set_xticks(range(len(Ws)))
            ax.set_xticklabels([str(w) for w in Ws])
            ax.set_xlabel("Network Width")

            # Y-axis label + ticks only on leftmost panel.
            if i == 0:
                ax.set_yticks(range(len(Ds)))
                ax.set_yticklabels([str(d) for d in Ds])
                ax.set_ylabel("Network Depth")
            else:
                ax.set_yticks([])
                ax.set_ylabel("")

            # Always show titles on the three-panel combined figure
            # (camera-ready spec); other heatmaps still honor paper_mode.
            if p.get("title"):
                ax.set_title(p["title"])

            # Cell annotations.
            for ii in range(len(Ds)):
                for jj in range(len(Ws)):
                    v = Z[ii, jj]
                    if np.isnan(v):
                        continue
                    se = SE[ii, jj]
                    if log_color:
                        mean_str, se_str = f"{v:.1e}", f"{se:.0e}"
                    else:
                        mean_str, se_str = f"{v:.2f}", f"{se:.1g}"
                    txt = mean_str if np.isnan(se) else f"{mean_str}\n$\\pm${se_str}"
                    ax.text(jj, ii, txt, ha="center", va="center",
                            color="white", fontsize=10 if paper_mode else 7)

            # Colorbar on every panel — ticks but no label (label in caption).
            fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05)

        # constrained_layout is set on the figure; no tight_layout needed.
        fig.savefig(save_path)
        plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir", type=str, required=True)
    parser.add_argument("--pivot_depth", type=int, default=4)
    parser.add_argument("--pivot_width", type=int, default=512)
    parser.add_argument("--metric", type=str, default="final_dyn_mse",
                        choices=["final_dyn_mse"])
    parser.add_argument("--paper_mode", action="store_true",
                        help="Drop titles + use paper fonts/sizes/dpi")
    parser.add_argument("--curve_palette", type=str, default="sequential",
                        choices=["sequential", "tab10"],
                        help="Slice-plot curve coloring: sequential (Blues / "
                             "Oranges) or tab10 (qualitative seaborn)")
    parser.add_argument("--cmap_heatmap", type=str, default=None,
                        help="Heatmap colormap (matplotlib/seaborn name). "
                             "Default = ARCH_CMAP_HEATMAP from plotting/style.py.")
    parser.add_argument("--wm_subdir", type=str, default="wm_track",
                        help="Per-cell WM-tracking subdir to load (default "
                             "'wm_track'; 'wm_track_policy' for policy variant).")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Override output dir (default: aggregate / "
                             "aggregate_<wm_subdir suffix>).")
    parser.add_argument("--depths", type=str, default=None,
                        help="Comma-separated depths to include (default: all).")
    parser.add_argument("--widths", type=str, default=None,
                        help="Comma-separated widths to include (default: all).")
    parser.add_argument("--unseen_return", type=str, default="disc",
                        choices=["disc", "undisc"],
                        help="Which return field to drive unseen-goal heatmaps "
                             "+ headlines: 'disc' (default) or 'undisc'.")
    parser.add_argument("--no_colorbar", action="store_true",
                        help="Drop the text label on heatmap colorbars (the "
                             "colorbar stays). Useful for paper figures that "
                             "describe the colorbar in the caption.")
    parser.add_argument("--log_pqn", action="store_true",
                        help="PQN return slice plots and heatmap on log scale "
                             "(matches WM-MSE). Off by default since returns "
                             "are O(1).")
    parser.add_argument("--wm_mse_ymax", type=float, default=None,
                        help="Cap WM-MSE slice y-axis + heatmap colorscale "
                             "(e.g. 1e-2). Clips bad-fit cells.")
    parser.add_argument("--unseen_only_labels", type=str, default=None,
                        help="Comma-separated unseen-goal labels to keep. "
                             "Default: all goals found in the npz files.")
    args = parser.parse_args()
    aggregate_arch_sweep(
        args.sweep_dir,
        pivot_depth=args.pivot_depth,
        pivot_width=args.pivot_width,
        save_dir=args.save_dir,
        metric=args.metric,
        paper_mode=args.paper_mode,
        curve_palette=args.curve_palette,
        cmap_heatmap=args.cmap_heatmap,
        wm_subdir=args.wm_subdir,
        depths_filter=(
            [int(x) for x in args.depths.split(",") if x] if args.depths else None
        ),
        widths_filter=(
            [int(x) for x in args.widths.split(",") if x] if args.widths else None
        ),
        unseen_return_kind=args.unseen_return,
        show_cbar=not args.no_colorbar,
        log_pqn=args.log_pqn,
        wm_mse_ymax=args.wm_mse_ymax,
        unseen_only_labels=(
            [s.strip() for s in args.unseen_only_labels.split(",") if s.strip()]
            if args.unseen_only_labels else None
        ),
    )
