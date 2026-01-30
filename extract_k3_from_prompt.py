# extract_k3_from_prompt.py
"""
Extract k=3 subset from prompt comparison CSV for paper figures.
Saves both the k=3 subset and a cleaned figure-specific CSV.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    in_csv = base_dir / "AnalyzeResults" / "abac_paper_PROMPT_comparison.csv"
    out_csv = base_dir / "AnalyzeResults" / "abac_paper_PROMPT_k3.csv"

    df = pd.read_csv(in_csv, low_memory=False)
    df.columns = df.columns.str.strip()

    if "k" in df.columns:
        df["k"] = pd.to_numeric(df["k"], errors="coerce")
        k3 = df[df["k"] == 3].copy()
    else:
        # fallback: match string '3'
        k3 = df[df["k"].astype(str) == "3"].copy()

    k3.to_csv(out_csv, index=False, na_rep="")
    print(f" Saved extracted data to {out_csv}")

    # --- Build paper figure CSV (k=3 specific filtering & cleaning) ---
    fig_out = base_dir / "AnalyzeResults" / "abac_paper_PROMPT_k3_fig.csv"

    # Read back the k3 file (robust column stripping already applied)
    df_k3 = pd.read_csv(out_csv, low_memory=False)
    df_k3.columns = df_k3.columns.str.strip()

    # Keep only the two FINAL attack classes shown in summary
    if "attack_class" in df_k3.columns:
        df_k3 = df_k3[df_k3["attack_class"].isin(["A_exfiltration_FINAL", "B_iterative_FINAL"])].copy()

    # If duplicates exist, keep the shortest output (or first) per key
    key = ["system", "k", "trial", "attack_class"]
    if "output_short" in df_k3.columns:
        df_k3["out_len"] = df_k3["output_short"].fillna("").astype(str).str.len()
        df_k3 = df_k3.sort_values("out_len").drop_duplicates(subset=key, keep="first")
    else:
        df_k3 = df_k3.drop_duplicates(subset=key, keep="first")

    # Optionally: pick trial 0 only for the paper figure
    if "trial" in df_k3.columns:
        fig = df_k3[pd.to_numeric(df_k3["trial"], errors="coerce").fillna(-1) == 0].copy()
    else:
        fig = df_k3.copy()

    fig.to_csv(fig_out, index=False, na_rep="")
    print(f"Wrote: {fig_out}")

if __name__ == "__main__":
    main()
