# analyze_results.py
"""Analyze ABAC CSV results

Reads CSVs from the Results folder, prints true record counts, fixes latency
columns, writes a flattened CSV for Excel, regenerates utility table with
latency, and exports a single Excel workbook with key tables.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _require_file(path: Path) -> None:
	if not path.exists():
		raise FileNotFoundError(f"Missing required file: {path}")


def _load_csv(path: Path) -> pd.DataFrame:
	_require_file(path)
	return pd.read_csv(path, low_memory=False)


def _compute_total_ms(df: pd.DataFrame) -> pd.DataFrame:
	for col in ["retrieval_s", "policy_s", "text_fetch_s", "gen_s"]:
		if col not in df.columns:
			df[col] = 0.0
	df["total_ms"] = (
		df["retrieval_s"].fillna(0)
		+ df["policy_s"].fillna(0)
		+ df["text_fetch_s"].fillna(0)
		+ df["gen_s"].fillna(0)
	) * 1000.0
	return df


def _flatten_newlines(df: pd.DataFrame) -> pd.DataFrame:
	out = df.copy()
	for col in ["question", "output"]:
		if col in out.columns:
			out[col] = (
				out[col]
				.fillna("")
				.astype(str)
				.str.replace(r"[\r\n\v\f\u2028\u2029]+", "\\n", regex=True)
			)
	return out


def _true_record_counts(df: pd.DataFrame) -> dict[str, int]:
	attack = df["attack_class"].astype(str)
	security_final = attack.str.endswith("_FINAL")
	counts = {
		"records": int(len(df)),
		"utility_rows": int((df["mode"] == "utility").sum()),
		"security_final_rows": int(security_final.sum()),
		"security_turn_rows": int((~security_final & df["attack_class"].notna()).sum()),
	}
	return counts


def _build_utility_table(df: pd.DataFrame) -> pd.DataFrame:
	util = df[df["mode"] == "utility"].copy()
	table = (
		util.groupby(["system", "k"]).agg(
			n=("EM", "count"),
			EM=("EM", "mean"),
			F1=("F1", "mean"),
			avg_effective_k=("effective_k", "mean"),
			p50_total_ms=("total_ms", "median"),
			p95_total_ms=("total_ms", lambda x: float(np.percentile(x, 95))),
		)
	).reset_index()
	return table


def main() -> None:
	base_dir = Path(__file__).resolve().parent
	abac_results_dir = base_dir / "Results"
	analyze_results_dir = base_dir / "AnalyzeResults"
	analyze_results_dir.mkdir(exist_ok=True)

	main_csv = abac_results_dir / "abac_rag_paper_aligned_results.csv"
	l1_csv = abac_results_dir / "table_L1_by_k.csv"
	leak_csv = abac_results_dir / "table_leakage_by_attack.csv"
	util_csv = abac_results_dir / "table_utility_by_k.csv"

	df = _load_csv(main_csv)
	df.columns = df.columns.str.strip()
	df = _compute_total_ms(df)

	print("len(df) =", len(df))
	print(df["mode"].value_counts(dropna=False))
	print("attack_class non-null =", df["attack_class"].notna().sum())
	print("FINAL rows =", df["attack_class"].fillna("").str.endswith("_FINAL").sum())

	counts = _true_record_counts(df)
	print("True record counts:")
	for k, v in counts.items():
		print(f"  {k}: {v}")

	flat_csv = analyze_results_dir / "abac_rag_paper_aligned_results_flat.csv"
	_flatten_newlines(df).to_csv(flat_csv, index=False, na_rep="")
	print(f"Wrote: {flat_csv}")

	util_table = _build_utility_table(df)
	util_out = analyze_results_dir / "table_utility_by_k_with_latency.csv"
	util_table.to_csv(util_out, index=False)
	print(f"Wrote: {util_out}")

	# Optional: load existing summary tables if present
	l1_df = _load_csv(l1_csv)
	leak_df = _load_csv(leak_csv)

	# Save summary tables as CSV (no Excel). Show previews as DataFrames.
	l1_out = analyze_results_dir / "table_L1_by_k_used.csv"
	leak_out = analyze_results_dir / "table_leakage_by_attack_used.csv"
	l1_df.to_csv(l1_out, index=False)
	leak_df.to_csv(leak_out, index=False)
	print(f"Wrote: {l1_out}")
	print(f"Wrote: {leak_out}")

	# Print concise previews (as DataFrame output)
	print("\nPreview (first 5 rows):")
	print("rows_flat:")
	print(_flatten_newlines(df).head().to_string(index=False))
	print("\nL1_by_k:")
	print(l1_df.head().to_string(index=False))
	print("\nleakage_by_attack:")
	print(leak_df.head().to_string(index=False))
	print("\nutility_by_k (regenerated with latency):")
	print(util_table.head().to_string(index=False))

	# Paper-friendly subset (utility + security FINAL only)
	paper_df = df[
		(df["mode"] == "utility")
		| (df["attack_class"].fillna("").str.endswith("_FINAL"))
	].copy()
	paper_out = analyze_results_dir / "abac_paper_subset.csv"
	_flatten_newlines(paper_df).to_csv(paper_out, index=False, na_rep="")
	print(f"Wrote: {paper_out}")

	# FINAL-only subset
	final_df = paper_df[paper_df["attack_class"].astype(str).str.endswith("_FINAL", na=False)].copy()
	final_out = analyze_results_dir / "abac_paper_subset_FINAL.csv"
	_flatten_newlines(final_df).to_csv(final_out, index=False, na_rep="")
	print(f"Wrote: {final_out}")
	print(len(final_df))
	print(final_df.head(5))


if __name__ == "__main__":
	main()
