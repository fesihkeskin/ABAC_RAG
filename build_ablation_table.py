# build_ablation_table.py
from pathlib import Path
import pandas as pd
import numpy as np

REFUSAL_PHRASE = "insufficient information in references"

def is_refusal_col(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.lower().str.contains(REFUSAL_PHRASE).astype(int)

def main():
    base = Path("AnalyzeResults")
    df = pd.read_csv(base / "abac_paper_subset.csv", low_memory=False)
    df.columns = df.columns.str.strip()

    # --- SECURITY: FINAL only ---
    sec = df[df["attack_class"].fillna("").astype(str).str.endswith("_FINAL")].copy()

    # --- UTILITY ---
    util = df[df["mode"].fillna("") == "utility"].copy()

    # If you log refused already, use it. Otherwise compute from output text.
    if "refused" in util.columns:
        util["refused"] = pd.to_numeric(util["refused"], errors="coerce").fillna(0).astype(int)
    else:
        # Requires that utility rows have "output"
        if "output" not in util.columns:
            raise RuntimeError("Utility rows do not contain 'output' or 'refused'. Please log one of them.")
        util["refused"] = is_refusal_col(util["output"])

    # For most EnronQA utility questions, refusal is treated as false refusal.
    util["false_refusal"] = util["refused"]

    # Map which variants to include
    variants = ["B3", "B3a", "B3b"]  # you will name systems accordingly in your run
    # Example mapping if you encode variants in system names:
    # B3-S2 = "B3_S2", B3a-S2 = "B3a_S2", B3b-S2 = "B3b_S2"
    # Update this list to your actual system strings:
    systems = ["B3_S2", "B3a_S2", "B3b_S2"]

    # --- Aggregate tables ---
    sec_out = (sec[sec["system"].isin(systems)]
               .groupby(["system", "k"], as_index=False)
               .agg(L1=("L1", "mean"),
                    L2=("L2_disclosure", "mean"),
                    L3=("L3_canary_leak", "mean")))

    util_out = (util[util["system"].isin([s + "_owner" for s in systems if not s.endswith("_owner")])]
                .groupby(["system", "k"], as_index=False)
                .agg(refusal=("refused", "mean"),
                     false_refusal=("false_refusal", "mean"),
                     F1=("F1", "mean")))

    # Normalize names so rows join cleanly
    util_out["system"] = util_out["system"].str.replace("_owner", "", regex=False)

    out = sec_out.merge(util_out, on=["system", "k"], how="left")
    out = out.sort_values(["system", "k"])

    out_path = base / "table_ablation_B3_variants.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
