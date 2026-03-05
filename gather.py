#!/usr/bin/env python3
"""Gather all results.json files from a runs directory, keyed by folder path."""

import argparse
import numpy as np
import json
from pathlib import Path
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty
from threading import Thread
from typing import Any, Dict, List, Optional, Type

BG = "#FFFFFF"
FG = "#1C1F24"
SPINE = "#444444"

TEAL = "#3A8FA3"
ORANGE = "#C46000"
GREEN = "#2A8A5E"
MAUVE = "#9E4A72"
VIOLET = "#6559A8"
RED = "#B83000"

PALETTE = [TEAL, ORANGE, GREEN, MAUVE, VIOLET, RED]
SERIF_STACK = ["Times New Roman", "Times", "DejaVu Serif", "serif"]


def apply_theme(
    context: str = "paper",
    font_scale: float = 1.35,
    palette: Optional[List[str]] = None,
    axes: Any = None,
) -> None:
    """
    Apply theme globally.

    Parameters
    ----------
    context    : seaborn context string
    font_scale : font scale multiplier
    palette    : list of hex colours; defaults to muted Jemoka palette
    axes       : Axes or array of Axes to label (a), (b), (c)...
    """

    import string
    import matplotlib as mpl
    import matplotlib.ticker as ticker
    import seaborn as sns
    import numpy as np

    if palette is None:
        palette = PALETTE

    sns.set_theme(
        context=context,
        style="ticks",
        palette=palette,
        font=SERIF_STACK[0],
        font_scale=font_scale,
        rc={
            "font.family": "serif",
            "font.serif": SERIF_STACK,
            "font.weight": "normal",
            "mathtext.fontset": "dejavuserif",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.titlesize": 10,
            "axes.labelsize": 9.5,
            "axes.titlecolor": FG,
            "axes.labelcolor": FG,
            "axes.titlepad": 9,
            "axes.labelpad": 5,
            "axes.linewidth": 0.9,
            "axes.edgecolor": SPINE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": BG,
            "axes.axisbelow": True,
            "axes.grid": True,
            "grid.color": "#D8D8D4",
            "grid.linewidth": 0.55,
            "grid.linestyle": "--",
            "grid.alpha": 0.65,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.top": False,
            "ytick.right": False,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.major.pad": 5,
            "ytick.major.pad": 5,
            "xtick.color": SPINE,
            "ytick.color": SPINE,
            "xtick.labelcolor": FG,
            "ytick.labelcolor": FG,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            # Key setting: limit ticks to a sensible maximum
            "axes.formatter.use_mathtext": True,
            "axes.formatter.limits": (-4, 4),
            "lines.linewidth": 1.7,
            "lines.markersize": 5.5,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#CCCCCA",
            "legend.fancybox": False,
            "legend.labelcolor": FG,
            "legend.fontsize": 8.5,
            "legend.borderpad": 0.4,
            "legend.labelspacing": 0.35,
            "legend.handlelength": 1.4,
            "legend.handletextpad": 0.5,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "patch.linewidth": 0.6,
        },
    )

    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=palette)

    # seaborn's categorical palette resolver checks rcParams["axes.prop_cycle"]
    # but hue-keyed plots fall back to their own defaults. Setting both
    # "palette" keys ensures consistency.
    sns.set_palette(palette)

    # Patch Axes.__init__ so every new axes gets MaxNLocator with
    # nbins="auto" — matplotlib will compute how many ticks fit without
    # overlapping given the actual axis length in display units.
    _orig_init = mpl.axes.Axes.__init__

    def _auto_tick_init(self: Any, *args: Any, **kwargs: Any) -> None:
        _orig_init(self, *args, **kwargs)
        self.xaxis.set_major_locator(
            ticker.MaxNLocator(nbins="auto", steps=[1, 2, 2.5, 5, 10], prune="both")
        )
        self.yaxis.set_major_locator(
            ticker.MaxNLocator(nbins="auto", steps=[1, 2, 2.5, 5, 10], prune="both")
        )

    mpl.axes.Axes.__init__ = _auto_tick_init

    if axes is not None:
        if isinstance(axes, mpl.axes.Axes):
            axes = [axes]
        else:
            axes = np.asarray(axes).ravel().tolist()
        for ax, label in zip(axes, string.ascii_lowercase):
            ax.text(
                -0.12,
                1.02,
                f"({label})",
                transform=ax.transAxes,
                fontsize=11,
                fontweight="normal",
                fontfamily="serif",
                va="bottom",
                ha="right",
                color=FG,
            )
apply_theme()

def gather_results(runs_dir: Path) -> dict:
    """Walk runs_dir and collect all results.json files.

    Keys are relative paths like 'celeba-64x64/VPCFM-Normal-16-factors'.
    """
    results = {}
    for results_file in sorted(runs_dir.rglob("results.json")):
        key = str(results_file.parent.relative_to(runs_dir))
        if "/" not in key:
            continue
        with open(results_file) as f:
            results[key] = json.load(f)
    return results


def stats(vals):
    """Compute mean and std, excluding zeros."""
    v = [x for x in vals if x > 0]
    if len(v) == 0:
        return 0.0, 0.0
    return np.mean(v), np.std(v)


def parse_entries(data):
    """Parse all entries from the raw data dict."""
    entries = []
    for key, val in data.items():
        ds, method_str = key.split("/")
        parts = method_str.split("-")
        flow = parts[0]
        base = parts[1]
        factors = int(parts[2])

        fid_m, fid_s = stats(val["FIDs"])
        ndb_m, ndb_s = stats(val["NDB/C"])
        nfe_m, nfe_s = stats(val["NFEs"])
        time_m, time_s = stats(val["total train times"])
        time_m /= 60.0
        time_s /= 60.0

        if fid_m == 0:
            continue

        entries.append({
            "ds": ds, "flow": flow, "base": base, "factors": factors,
            "fid_m": fid_m, "fid_s": fid_s,
            "ndb_m": ndb_m, "ndb_s": ndb_s,
            "nfe_m": nfe_m, "nfe_s": nfe_s,
            "time_m": time_m, "time_s": time_s,
        })
    return entries


def fmt_metric(m, s, best, prec=1):
    if m == 0:
        return "---"
    is_best = abs(m - best) < 1e-6
    if is_best:
        return f"\\bfseries {m:.{prec}f} \\pm {s:.{prec}f}"
    return f"{m:.{prec}f} \\pm {s:.{prec}f}"


def main_table(all_entries):
    """Table 1: Main comparison across datasets.

    For each dataset, show VPCFM/OTCFM x MPPCA/Normal at the canonical L.
    """
    dataset_info = {
        "fashion":      {"cmd": r"\fashion{}",  "dims": r"28 \times 28 \times 1", "epochs": 100},
        "cifar10":      {"cmd": r"\cifar{}",    "dims": r"32 \times 32 \times 3", "epochs": 100},
        "celeba":       {"cmd": r"\celeba{}",  "dims": r"32 \times 32 \times 3", "epochs": 50},
        "celeba-64x64": {"cmd": r"\celeba{}", "dims": r"64 \times 64 \times 3", "epochs": 50},
    }
    dataset_order = ["fashion", "cifar10", "celeba", "celeba-64x64"]
    # Canonical L per dataset
    canonical_l = {"fashion": 6, "cifar10": 10, "celeba": 10, "celeba-64x64": 16}

    # Model display name: VP-MPPCA, OT-MPPCA, VP-Normal, OT-Normal
    def model_name(e):
        prefix = "VP" if e["flow"] == "VPCFM" else "OT"
        return f"{prefix}-{e['base']}"

    # Sort: VP-MPPCA, OT-MPPCA, VP-Normal, OT-Normal
    def sort_key(e):
        return (0 if e["base"] == "MPPCA" else 1,
                0 if e["flow"] == "VPCFM" else 1)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Generative modeling results on image datasets.}")
    lines.append(r"\label{tab:images}")
    lines.append(r"\small")
    lines.append(r"    \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}")
    lines.append(r"        c")
    lines.append(r"        l")
    lines.append(r"        r")
    lines.append(r"        S[table-format=3.1(1.1)]")
    lines.append(r"        S[table-format=2.2(1.2)]")
    lines.append(r"        S[table-format=3.1(1.1)]")
    lines.append(r"        S[table-format=3.1(1.1)]")
    lines.append(r"    @{}}")
    lines.append(r"    \toprule")
    lines.append(r"     \multirow{2}{*}{\vspace{-5pt}Dataset} & \multirow{2}{*}{\vspace{-5pt}Model} & \multicolumn{2}{c}{Training} & \multicolumn{3}{c}{Testing} \\")
    lines.append(r"    \cmidrule(lr){3-4} \cmidrule(lr){5-7}")
    lines.append(r"     &   & {epochs} & {time (min)} & {NDB/$C$} & {NFE} & {FID}\\")
    lines.append(r"    \midrule")

    active_datasets = [ds for ds in dataset_order
                       if any(e["ds"] == ds and e["factors"] == canonical_l[ds]
                              for e in all_entries)]

    for di, ds in enumerate(active_datasets):
        info = dataset_info[ds]
        l = canonical_l[ds]
        entries = [e for e in all_entries if e["ds"] == ds and e["factors"] == l]
        entries.sort(key=sort_key)

        best_ndb = min(e["ndb_m"] for e in entries)
        best_nfe = min(e["nfe_m"] for e in entries)
        best_fid = min(e["fid_m"] for e in entries)

        n = len(entries)
        ds_label = (
            f"\\multirow{{{n}}}{{*}}"
            f"{{\\shortstack[*]{{\\textbf{{{info['cmd']}}}\\\\$[{info['dims']}]$}}}}"
        )

        for i, e in enumerate(entries):
            ndb_str = fmt_metric(e["ndb_m"], e["ndb_s"], best_ndb, 2)
            nfe_str = fmt_metric(e["nfe_m"], e["nfe_s"], best_nfe, 1)
            fid_str = fmt_metric(e["fid_m"], e["fid_s"], best_fid, 1)
            time_str = f"{e['time_m']:.1f} \\pm {e['time_s']:.1f}"

            row = (
                f"        & {model_name(e)}  "
                f"& {info['epochs']}   "
                f"& {time_str} "
                f"& {ndb_str} "
                f"& {nfe_str} "
                f"& {fid_str} \\\\"
            )
            if i == 0:
                lines.append(f"    {ds_label}")
            lines.append(row)

        if di < len(active_datasets) - 1:
            lines.append(r"    \midrule")

    lines.append(r"    \bottomrule")
    lines.append(r"    \end{tabular*}")
    lines.append(r"\end{table*}%")
    return "\n".join(lines)


def sweep_table(all_entries):
    """Table 2: Fashion-MNIST VPCFM-MPPCA L sweep, with OTCFM-MPPCA baseline."""
    sweep = [e for e in all_entries
             if e["ds"] == "fashion" and e["base"] == "MPPCA"]

    # Sort: OTCFM first (baseline), then VPCFM by K
    def sort_key(e):
        return (0 if e["flow"] == "OTCFM" else 1, e["factors"])
    sweep.sort(key=sort_key)

    best_fid = min(e["fid_m"] for e in sweep)
    best_ndb = min(e["ndb_m"] for e in sweep)
    best_nfe = min(e["nfe_m"] for e in sweep)
    best_time = min(e["time_m"] for e in sweep)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Effect of the number of MPPCA components $L$ on Fashion-MNIST. Best results in \textbf{bold}.}")
    lines.append(r"\label{tab:sweep}")
    lines.append(r"\begin{tabular}{ll r c ccc}")
    lines.append(r"\toprule")
    lines.append(r" & & & \multicolumn{1}{c}{\textbf{Training}} & \multicolumn{3}{c}{\textbf{Testing}} \\")
    lines.append(r"\cmidrule(lr){4-4} \cmidrule(lr){5-7}")
    lines.append(r"Flow & Base & $L$ & Time (min) & NDB/C $\downarrow$ & NFE $\downarrow$ & FID $\downarrow$ \\")
    lines.append(r"\midrule")

    wrote_vpcfm = False
    for e in sweep:
        if e["flow"] == "VPCFM" and not wrote_vpcfm:
            lines.append(r"\hdashline\noalign{\vskip 2pt}")
            wrote_vpcfm = True

        fid_str = fmt_metric(e["fid_m"], e["fid_s"], best_fid, 1)
        ndb_str = fmt_metric(e["ndb_m"], e["ndb_s"], best_ndb, 2)
        nfe_str = fmt_metric(e["nfe_m"], e["nfe_s"], best_nfe, 1)
        time_str = fmt_metric(e["time_m"], e["time_s"], best_time, 1)

        lines.append(
            f"  {e['flow']} & {e['base']} & ${e['factors']}$ "
            f"& {time_str} & {ndb_str} & {nfe_str} & {fid_str} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

def plot_sweep(all_entries, path: str = "sweep"):
    """Three separate sweep plots: NFE, NDB/C, FID vs L for Fashion-MNIST."""
    import matplotlib.pyplot as plt
    apply_theme()

    sweep = pd.DataFrame([
        e for e in all_entries
        if e["ds"] == "fashion" and e["base"] == "MPPCA" and e["flow"] == "VPCFM"
    ]).sort_values("factors")

    stem = Path(path).stem
    suffix = Path(path).suffix or ".pdf"
    parent = Path(path).parent

    panels = [
        ("nfe_m",  "nfe_s",  "NFE",        f"{stem}_nfe{suffix}",  TEAL),
        ("ndb_m",  "ndb_s",  "NDB/$C$",    f"{stem}_ndb{suffix}", ORANGE),
        ("fid_m",  "fid_s",  "FID",        f"{stem}_fid{suffix}", GREEN),
    ]

    for yk, sk, ylabel, fname, color in panels:
        fig, ax = plt.subplots(1, 1, figsize=(4.0, 2.2))
        x = sweep["factors"].values
        y = sweep[yk].values
        s = sweep[sk].values
        ax.plot(x, y, marker="o", color=color)
        ax.fill_between(x, y - s, y + s, alpha=0.18, color=color, linewidth=0)
        ax.set_xlabel("$\\ell$")
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        out = parent / fname
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")

def main():
    parser = argparse.ArgumentParser(description="Gather results.json files from a runs directory.")
    parser.add_argument("runs_dir", nargs="?", default="./runs",
                        help="Path to runs directory (default: ./runs)")
    parser.add_argument("-o", "--output",
                        help="Write gathered results to this JSON file")
    parser.add_argument("-t", "--latex",
                        help="Write main LaTeX table to this file")
    parser.add_argument("-s", "--sweep-latex",
                        help="Write sweep LaTeX table to this file")
    parser.add_argument("-p", "--plot",
                        help="Write sweep plot to this PDF/PNG file")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        raise SystemExit(f"Error: '{runs_dir}' is not a directory.")

    results = gather_results(Path("./runs"))
    print(f"Found {len(results)} result(s).", flush=True)

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))

    all_entries = parse_entries(results)

    if args.latex:
        Path(args.latex).write_text(main_table(all_entries))
        print(f"Wrote main table to {args.latex}")

    if args.sweep_latex:
        Path(args.sweep_latex).write_text(sweep_table(all_entries))
        print(f"Wrote sweep table to {args.sweep_latex}")

    if args.plot:
        plot_sweep(all_entries, path=args.plot)


if __name__ == "__main__":
    main()
