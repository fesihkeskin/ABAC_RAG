#!/bin/bash
#SBATCH -p akya-cuda              # ARF nodes with V100 GPUs
#SBATCH -A fekeskin               # Your account name
#SBATCH -J abac_rag               # Job name
#SBATCH -o abac_result_%J.out     # Output file (%j appends the job ID)
#SBATCH -e abac_error_%J.err      # Error file
#SBATCH -N 1                      # Single node
#SBATCH -n 1                      # Single task
#SBATCH --cpus-per-task=40        # Multiple cores for FAISS and data processing (Parallel CPU)
#SBATCH --mem=350G                # Sufficient RAM for 8B Model + Dataset
#SBATCH --gres=gpu:4              # 1 GPU for Llama-3.1
#SBATCH --time=72:00:00           # Duration based on experiment scope (Max 72h recommended)

echo "Job has started: $(date)"
echo "Hostname: $(hostname)"
# 1. File paths and Cache settings per TRUBA regulations
# Redirecting to Scratch to prevent models from filling the Home directory
export HF_HOME=/arf/scratch/fekeskin/hf_cache
export TRANSFORMERS_CACHE=/arf/scratch/fekeskin/hf_cache/transformers
export HF_DATASETS_CACHE=/arf/scratch/fekeskin/hf_cache/datasets

# Ensure the directories exist
mkdir -p $HF_HOME
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_DATASETS_CACHE

# Queue monitoring (excluding own job)
QUEUE_CHECK_INTERVAL_MIN=2
QUEUE_USER="${USER}"
JOB_ID="${SLURM_JOB_ID}"


# 2. Activate Conda Environment
eval "$(/arf/home/fekeskin/miniconda3/bin/conda shell.bash hook)"
conda activate pytorchgpu_env

# 3. Start Experiment (with CLI Parameters)
# Note: Initial test values like max_email_rows are kept low; 
# these can be increased once you confirm everything is running correctly.

echo "GPU test starting..."
python -c "import torch; print('GPU working:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"

echo "ABAC RAG experiment starting..."
# print parameters used
echo "Experiment parameters:"
echo "max_email_rows: 10000"
echo "max_qa_utility: 300"
echo "max_qa_security: 600"
echo "n_trials: 3"
echo "checkpoint_dir: /arf/scratch/fekeskin/abac/checkpoints"
echo "hf_cache_dir: /arf/scratch/fekeskin/hf_cache"
echo "sample_n_tuning: 200" 

python abac_rag.py \
    --max_email_rows 10000 \
    --max_qa_utility 300 \
    --max_qa_security 600 \
    --n_trials 3 \
    --checkpoint_dir /arf/scratch/fekeskin/abac/checkpoints \
    --hf_cache_dir /arf/scratch/fekeskin/hf_cache \
    --sample_n_tuning 200
# --local_files_only  # to read from local cache only


# 4. Post-Job Notification
echo "ABAC RAG experiment completed."

# Usage: sbatch run_abac_exp.slurm