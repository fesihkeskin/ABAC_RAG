# find_final_b1_examples.py
"""
Find final B1 examples from the full paper subset CSV.
Saves flagged examples and a prompt comparison CSV for paper figures.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

def main() -> None:
    base_dir = Path(__file__).resolve().parent
    final_csv = base_dir / "AnalyzeResults" / "abac_paper_subset_FINAL.csv"
    out_dir = base_dir / "AnalyzeResults"
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv(final_csv, low_memory=False)
    df.columns = df.columns.str.strip()

    # make key numeric columns robust
    num_cols = ["canary_exposure","L2_disclosure","L3_canary_leak","inj_compliance",
                "prefilter_text_exposed","L1","effective_k","total_ms","cand_denied","fetch_k","turn"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # focus on B1 FINAL rows
    b1 = df[df["system"] == "B1"].copy()
    b1_final = b1[b1["attack_class"].astype(str).str.endswith("_FINAL")].copy()

    # flagged examples (canary exposure or L2 disclosure)
    b1_flagged = b1_final[(b1_final["canary_exposure"] >= 1) | (b1_final["L2_disclosure"] >= 1)].copy()
    examples = b1_flagged.sort_values(
        ["canary_exposure","L3_canary_leak","inj_compliance","L2_disclosure","total_ms"],
        ascending=False
    ).head(10)

    flagged_out = out_dir / "abac_paper_FINAL_B1_flagged_examples.csv"
    examples.to_csv(flagged_out, index=False, na_rep="")
    print(f"Wrote: {flagged_out}")

    # Choose the best prompt for the paper figure
    # Prefer canary_exposure=1, else fall back to L2
    if not b1_flagged.empty:
        cand = b1_flagged.copy()
    else:
        cand = b1_final.copy()

    cand["score"] = (
        20*cand["L3_canary_leak"]
        + 20*cand["inj_compliance"]
        + 10*cand["canary_exposure"]
        + 5*cand["L2_disclosure"]
        + 0.1*cand["total_ms"]
        + cand["k"].fillna(0)  # slight preference for larger k
    )
    best = cand.sort_values("score", ascending=False).iloc[0]
    prompt = best["question"]
    print("Selected prompt:", prompt)

    systems_to_show = ["B1","B2","B2u","B3_S1","B3_S2"]
    subset = df[(df["question"] == prompt) & (df["system"].isin(systems_to_show))].copy()

    # paper-friendly output fields
    subset["output"] = subset["output"].fillna("").astype(str)
    subset["output_flat"] = subset["output"].str.replace(r"[\r\n]+", "\\n", regex=True)
    subset["output_short"] = subset["output_flat"].str.slice(0, 450)

    cols = [
        "system","k","trial","attack_class","turn",
        "effective_k","fetch_k",
        "L1","prefilter_text_exposed","canary_exposure",
        "L2_disclosure","L3_canary_leak","inj_compliance",
        "cand_denied","total_ms","output_short"
    ]
    comparison_out = out_dir / "abac_paper_PROMPT_comparison.csv"
    subset[cols].sort_values(["system","k","trial"]).to_csv(comparison_out, index=False, na_rep="")
    print(f"Wrote: {comparison_out}")

    # aggregated summary table (one row per system/k/attack_class for this prompt)
    summary = subset.groupby(["system","k","attack_class"], as_index=False).agg(
        n=("trial","count"),
        p50_total_ms=("total_ms","median"),
        effk_med=("effective_k","median"),
        L1_rate=("L1","mean"),
        prefilter_rate=("prefilter_text_exposed","mean"),
        canary_rate=("canary_exposure","mean"),
        L2_rate=("L2_disclosure","mean"),
        L3_rate=("L3_canary_leak","mean"),
        inj_rate=("inj_compliance","mean"),
        cand_denied_med=("cand_denied","median"),
    )
    summary_out = out_dir / "abac_paper_PROMPT_comparison_summary.csv"
    summary.to_csv(summary_out, index=False, na_rep="")
    print(f"Wrote: {summary_out}")

if __name__ == "__main__":
    main()
