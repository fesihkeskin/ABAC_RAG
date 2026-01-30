ABAC-Enforced Retrieval for Secure Multi-Tenant RAG
===================================================

This repository contains the code, data-processing utilities, and paper artifacts for the study on policy-bound retrieval with Attribute-Based Access Control (ABAC) in Retrieval-Augmented Generation (RAG). The main experiment implements multiple retrieval systems (global, post-filter baselines, and ABAC-enforced retrieval), evaluates security leakage under adversarial prompts, and reports utility/latency trade-offs.

Architecture
------------

The system enforces ABAC at retrieval time by separating policy decision (PDP) from enforcement (PEP). The retriever acts as the PEP and consults the PDP before candidates can enter the prompt context.

See the architecture diagram in [abac_rag_architecture.png](abac_rag_architecture.png).

Repository Contents
-------------------

- Main experiment: [abac_rag.py](abac_rag.py)
- Analysis and table generation: [analyze_results.py](analyze_results.py)
- Paper figure generation: [plot_paper_figures.py](plot_paper_figures.py)
- Paper artifact bundle (alternative entry): [build_paper_artifacts.py](build_paper_artifacts.py)
- Ablation table builder: [build_ablation_table.py](build_ablation_table.py)
- Example selection for prompt comparison: [find_final_b1_examples.py](find_final_b1_examples.py)
- Prompt k=3 extractor for figures: [extract_k3_from_prompt.py](extract_k3_from_prompt.py)
- Post-hoc guard for B3b security results: [posthoc_guard_b3b.py](posthoc_guard_b3b.py)
- Paper LaTeX template: [cas-sc-template.tex](cas-sc-template.tex)

Summary of Methods
------------------

- **Policy-bound retrieval:** ABAC policies are enforced during retrieval so unauthorized chunks are never eligible for inclusion in the prompt context.
- **Attack suites:** Cross-tenant exfiltration, iterative extraction, and indirect prompt injection.
- **Leakage metrics:**
  - **L1**: unauthorized chunks appearing in retrieved top-$k$.
  - **L2**: unauthorized disclosure based on text overlap and semantic similarity.
  - **L3**: canary leakage (exact canary strings appearing in outputs).
- **Utility metrics:** EM/F1 for QA tasks on authorized data.
- **Latency:** aggregated retrieval, policy, fetch, and generation times.

## Data and Experimental Setup

We evaluate on an email-domain proxy for multi-user private retrieval derived from the public Enron email corpus. The Enron corpus contains emails organized into folders for approximately 150 users and has been widely used as a public research resource; it was originally made public during the Federal Energy Regulatory Commission (FERC) investigation and later curated for research distribution.

Our workload follows the EnronQA benchmark formulation, which constructs question–answer (QA) annotations over Enron emails to support retrieval and grounded QA in a multi-user setting. The EnronQA paper reports 150 inboxes and large-scale QA annotations over Enron emails, enabling controlled evaluation of cross-inbox (cross-tenant) access attempts.

Concretely, we use the Hugging Face release `MichaelR207/enron_qa_0922`, which packages email-level examples with QA annotations (e.g., email text, sender/user identifier, file path, and lists of questions and gold answers), and provides train/dev/test splits. Each example corresponds to a single email instance augmented with multiple questions and gold answers, enabling (i) authorized QA evaluation when the querying user matches the email owner, and (ii) multi-tenant security evaluation by treating inbox owners as tenants.

The emails reflect real-world corporate communication (scheduling, operational planning, policy, finance, and other organizational topics) and contain domain-specific language. Since the underlying corpus contains real communications, we report results in aggregate and keep qualitative excerpts short and non-identifying.

We follow the dataset-provided train/dev/test splits and use compute-bounded sampling to enable full end-to-end evaluation with a local 8B generator. We construct the retrieval corpus from a subset of the training emails, evaluate utility on a fixed set of test QA pairs, and evaluate security on a template-instantiated adversarial prompt suite across three attack classes. Each condition is evaluated with three independent decoding trials.

### Experimental Configuration Summary

| Component | Setting |
|---|---|
| Dataset | EnronQA (`enron_qa_0922`; train/dev/test splits) |
| Corpus used for indexing | 10,000 emails sampled from the training split |
| Tenants (security eval.) | Fixed attacker–victim pair (distinct inbox owners); attacker is low-privilege (low clearance, no cross-tenant sharing) |
| Chunking | 256 tokens per chunk, 50-token overlap |
| Chunk tokenizer | `bert-base-uncased` |
| Embedding model | `all-MiniLM-L6-v2` |
| Vector index | FAISS exact IP over normalized vectors (`IndexFlatIP` + ID map) |
| Generator | `Meta-Llama-3.1-8B-Instruct` (4-bit quantized) |
| Decoding | Temperature 0.1; maximum 512 new tokens; 3 decoding trials per condition |
| Top-$k$ values | $k \in \{3,5,10\}$ |
| Over-fetch for S2 | $fetch_k = 100$ (ABAC filtering may yield $effective_k < k$ under strict policy) |
| Utility workload size | 300 QA pairs (test split); 3 trials each ⇒ 900 generations per system/$k$ |
| Security workload size | 600 instantiated prompts; 3 trials each; B uses 3 turns (+ one *Final* aggregation) |
| Attack classes | A: exfiltration; B: iterative extraction; C: indirect injection |
| Reporting unit (security) | *Final* rows (one aggregated output per prompt/trial) |
| Compute platform | TRUBA (TUBITAK ULAKBIM); 1 node, Tesla V100 (16GB), 40 CPU cores allocated |
| Total records | 82,800 (10,800 utility; 72,000 security incl. per-turn + *Final*) |

Emails are segmented into fixed-length token windows using 256-token chunks with 50-token overlap. Chunking uses a BERT-family tokenizer (`bert-base-uncased`) for stable tokenization across heterogeneous email text. Each chunk retains provenance fields (document ID, thread ID, folder) and the metadata needed for authorization.

Dense embeddings are computed using `all-MiniLM-L6-v2` (SentenceTransformers) and indexed in FAISS using an inner-product index over normalized vectors (cosine similarity), implemented as `IndexFlatIP` wrapped by `IndexIDMap2`. The global index enables baseline retrieval; ABAC-enforced variants additionally route queries through authorized partitions (S1) or over-fetch and filter (S2) prior to context assembly.

Each chunk stores the attributes required by ABAC enforcement, minimally `owner_tenant` (mapped from the Enron inbox owner), `folder`, and `sensitivity`. Sensitivity labels are derived using a keyword-based heuristic and are used for clearance-gated retrieval in our ABAC policy. Tenant isolation is enforced for all systems that implement authorization.

Key Results (Paper-Aligned)
---------------------------

- ABAC-enforced retrieval eliminates policy-violating retrieval ($L1=0$ for $k\in\{3,5,10\}$) in the paper-aligned runs.
- Global retrieval and post-filtering baselines exhibit non-zero policy violations and higher leakage under adversarial prompting.
- Utility is preserved with comparable EM/F1 on authorized QA; latency remains in the low-second range for typical settings.

These values are produced from the CSVs in [Results/](Results/) and the derived tables in [AnalyzeResults/](AnalyzeResults/).

Reproducibility
---------------

The pipeline supports both **local CLI** and **Slurm** execution.

## Prerequisites:

- Python environment with PyTorch, transformers, sentence-transformers, datasets, faiss, numpy, pandas, tqdm.
- GPU recommended for local runs (the default model is an 8B LLM in 4-bit).

### Run Order (Reproducibility)

1. Run the main experiment to generate raw results in [Results/](Results/):
	- [abac_rag.py](abac_rag.py)
2. Analyze results and regenerate tables in [AnalyzeResults/](AnalyzeResults/):
	- [analyze_results.py](analyze_results.py)
3. (Optional) Build ablation table variants:
	- [build_ablation_table.py](build_ablation_table.py)
4. (Optional) Generate prompt comparison artifacts:
	- [find_final_b1_examples.py](find_final_b1_examples.py)
	- [extract_k3_from_prompt.py](extract_k3_from_prompt.py)
5. Generate figures for the paper bundle in [PaperArtifacts/](PaperArtifacts/):
	- [plot_paper_figures.py](plot_paper_figures.py)

### Local CLI:

1. Run the main experiment:

	- Entry point: [abac_rag.py](abac_rag.py)
	- Typical arguments:
	  - ``--max_email_rows`` (limit corpus size for quick tests)
	  - ``--max_qa_utility`` and ``--max_qa_security``
	  - ``--n_trials``
	  - ``--systems`` (comma list, e.g., ``B1,B2,B2u,B3_S1,B3_S2``)
	  - ``--k_list`` (e.g., ``3,5,10``)
	  - ``--checkpoint_dir`` and ``--hf_cache_dir``

2. Analyze results and build tables:

	- [analyze_results.py](analyze_results.py)
	- Outputs to [AnalyzeResults/](AnalyzeResults/)

3. Generate paper figures:

	- [plot_paper_figures.py](plot_paper_figures.py)
	- Outputs to [PaperArtifacts/](PaperArtifacts/)

### Slurm Execution

Use the provided Slurm script for cluster execution:

- [run_abac_exp.slurm](run_abac_exp.slurm)

Typical workflow:

1. Edit Slurm parameters (partition, GPU, time, and environment activation).
2. Submit with ``sbatch``.
3. Collect outputs from [Results/](Results/) and re-run analysis/plots locally or on the cluster.

### Output Artifacts
----------------

After running the pipeline, key outputs include:

- [Results/abac_rag_paper_aligned_results.csv](Results/abac_rag_paper_aligned_results.csv)
- [AnalyzeResults/abac_rag_paper_aligned_results_flat.csv](AnalyzeResults/abac_rag_paper_aligned_results_flat.csv)
- [AnalyzeResults/table_L1_by_k_used.csv](AnalyzeResults/table_L1_by_k_used.csv)
- [AnalyzeResults/table_leakage_by_attack_used.csv](AnalyzeResults/table_leakage_by_attack_used.csv)
- [AnalyzeResults/table_utility_by_k_with_latency.csv](AnalyzeResults/table_utility_by_k_with_latency.csv)

### Notes
-----

- The experiment uses the EnronQA-derived dataset specified in [abac_rag.py](abac_rag.py).
- Canary strings are injected into a subset of confidential chunks to precisely measure leakage.
- Post-filter baselines can still incur pre-filter exposure and evidence starvation, which is explicitly measured.

## Citation

If you find this work useful in your research, please consider citing:

```BibTeX
@Article{Oz2026,
  author   = {Gulser Oz and Fesih Keskin},
  journal  = {},
  title    = {},
  year     = {2026},
  issn     = {},
  pages    = {},
  doi      = {},
  keywords = {},
  url      = {},
  note   = {Manuscript in preparation.},
}

```
###  License
-------
This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Acknowledgments
---------------

Numerical calculations were performed at TUBITAK ULAKBIM High Performance and Grid Computing Center ([TRUBA](https://www.truba.gov.tr/index.php/en/main-page/)). We also thank the creators of the EnronQA benchmark for enabling multi-user email retrieval research.

---
For questions or contributions, please open an issue or pull request!

---

