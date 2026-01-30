# build_paper_figures.py
"""
Build paper figures from analysis CSVs.
Generates line plots for L1 rate vs k, L2 disclosure by attack, and latency.
Saves plots as PNG and EPS files in the output directory.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import matplotlib.pyplot as plt


ARTIFACT_FILES: List[str] = []


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _lineplot_by_system(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    system_col: str = "system",
    systems: Optional[Iterable[str]] = None,
    ax: Optional[plt.Axes] = None,
    ylabel: Optional[str] = None,
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()

    if systems is None:
        systems = sorted(df[system_col].dropna().unique().tolist())

    for system in systems:
        sdf = df[df[system_col] == system].copy()
        if sdf.empty:
            continue
        sdf = sdf.sort_values(x_col)
        ax.plot(sdf[x_col], sdf[y_col], marker="o", label=str(system))

    ax.set_xlabel(x_col)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return ax


def _save_plot(fig: plt.Figure, out_base: Path) -> List[str]:
    outputs = []
    png_path = out_base.with_suffix(".png")
    eps_path = out_base.with_suffix(".eps")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(eps_path, bbox_inches="tight")
    outputs.extend([png_path.name, eps_path.name])
    return outputs


def _plot_l1_vs_k(df: pd.DataFrame, out_base: Path) -> List[str]:
    df = df.copy()
    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    y_col = "L1_event_rate" if "L1_event_rate" in df.columns else "L1_rate"
    systems = ["B1", "B2", "B2u", "B3_S1", "B3_S2"]

    fig = plt.figure(figsize=(6.5, 4.0))
    _lineplot_by_system(df, "k", y_col, systems=systems, ylabel="L1 rate")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    outputs = _save_plot(fig, out_base)
    plt.close(fig)
    return outputs


def _plot_l2_by_attack(df: pd.DataFrame, out_base: Path) -> List[str]:
    df = df.copy()
    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    y_col = "L2_disclosure_rate" if "L2_disclosure_rate" in df.columns else "L2_rate"
    attack_classes = sorted(df["attack_class"].dropna().unique().tolist())
    systems = ["B1", "B2", "B2u", "B3_S1", "B3_S2"]

    fig, axes = plt.subplots(1, len(attack_classes), figsize=(12.5, 3.6), sharey=True)
    if len(attack_classes) == 1:
        axes = [axes]

    for ax, attack in zip(axes, attack_classes):
        sdf = df[df["attack_class"] == attack]
        _lineplot_by_system(sdf, "k", y_col, systems=systems, ax=ax)
        ax.set_title(str(attack))

    axes[0].set_ylabel("L2 disclosure rate")
    axes[-1].legend(ncol=1, fontsize=8, loc="best")
    fig.tight_layout()
    outputs = _save_plot(fig, out_base)
    plt.close(fig)
    return outputs


def _plot_latency(
    df: pd.DataFrame,
    out_base: Path,
    title: str,
    systems: Optional[Iterable[str]] = None,
) -> List[str]:
    df = df.copy()
    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    p50 = "p50_total_ms" if "p50_total_ms" in df.columns else "P50"
    p95 = "p95_total_ms" if "p95_total_ms" in df.columns else "P95"

    if systems is None:
        systems = sorted(df["system"].dropna().unique().tolist())

    fig = plt.figure(figsize=(6.0, 4.0))
    ax = fig.gca()
    for system in systems:
        sdf = df[df["system"] == system].sort_values("k")
        if sdf.empty:
            continue
        ax.plot(sdf["k"], sdf[p50], marker="o", label=f"{system} P50")
        ax.plot(sdf["k"], sdf[p95], marker="x", linestyle="--", label=f"{system} P95")

    ax.set_title(title)
    ax.set_xlabel("k")
    ax.set_ylabel("Latency (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    outputs = _save_plot(fig, out_base)
    plt.close(fig)
    return outputs


def _generate_figures(analyze_dir: Path, out_dir: Path) -> List[str]:
    generated: List[str] = []

    l1_csv = analyze_dir / "table_L1_by_k_used.csv"
    l2_csv = analyze_dir / "table_leakage_by_attack_used.csv"
    util_csv = analyze_dir / "table_utility_by_k_with_latency.csv"

    if l1_csv.exists():
        df_l1 = pd.read_csv(l1_csv)
        out_base = out_dir / "fig_L1_rate_vs_k"
        generated.extend(_plot_l1_vs_k(df_l1, out_base))

    if l2_csv.exists():
        df_l2 = pd.read_csv(l2_csv)
        out_base = out_dir / "fig_L2_by_attack"
        generated.extend(_plot_l2_by_attack(df_l2, out_base))

    if l1_csv.exists():
        df_l1 = pd.read_csv(l1_csv)
        out_base = out_dir / "fig_latency_security"
        generated.extend(
            _plot_latency(
                df_l1,
                out_base,
                title="Security latency (P50/P95)",
                systems=["B1", "B2", "B2u", "B3_S1", "B3_S2"],
            )
        )

    if util_csv.exists():
        df_util = pd.read_csv(util_csv)
        out_base = out_dir / "fig_latency_utility"
        generated.extend(
            _plot_latency(
                df_util,
                out_base,
                title="Utility latency (P50/P95)",
                systems=["B1_owner", "B2_owner", "B3_S1_owner", "B3_S2_owner"],
            )
        )

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PaperArtifacts bundle and figures.")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--analyze-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    base_dir = args.base_dir
    analyze_dir = args.analyze_dir or (base_dir / "AnalyzeResults")
    out_dir = args.out_dir or (base_dir / "PaperArtifacts")

    _ensure_dir(out_dir)
    generated = _generate_figures(analyze_dir, out_dir)

    print(f"PaperArtifacts: {out_dir}")
    print(f"Figures ({len(generated)}): {', '.join(generated)}")


if __name__ == "__main__":
    main()
