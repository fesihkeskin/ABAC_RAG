# posthoc_guard_b3b.py
"""
Apply post-hoc guard to B3b security FINAL results.
Saves modified CSV with guard applied to L2 and L3 metrics and outputs.
"""

from pathlib import Path
import pandas as pd

REFUSAL = "Insufficient information in references."

def main():
    base = Path("AnalyzeResults")
    df = pd.read_csv(base / "abac_paper_subset.csv", low_memory=False)
    df.columns = df.columns.str.strip()

    # Only security FINAL rows (you can also do per-turn if you want)
    sec = df[df["attack_class"].fillna("").astype(str).str.endswith("_FINAL")].copy()

    # Apply guard post hoc
    sec["guard_triggered"] = ((sec["L2_disclosure"] == 1) | (sec["L3_canary_leak"] == 1)).astype(int)
    sec["L2_after_guard"] = sec["L2_disclosure"] * 0
    sec["L3_after_guard"] = sec["L3_canary_leak"] * 0
    sec["output_after_guard"] = sec["output"]
    sec.loc[sec["guard_triggered"] == 1, "output_after_guard"] = REFUSAL

    out = base / "abac_security_FINAL_with_posthoc_guard.csv"
    sec.to_csv(out, index=False)
    print(f"Wrote: {out}")

if __name__ == "__main__":
    main()
# usage: python posthoc_guard_b3b.py