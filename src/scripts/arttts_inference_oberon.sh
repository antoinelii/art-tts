#!/bin/bash
#SBATCH --job-name=arttts_infer     # Job name
#SBATCH --partition=gpu              # Take a node from the 'gpu' partition
#SBATCH --export=ALL                  # Export your environment to the compute node
#SBATCH --cpus-per-task=1             # Ask for 6 CPU cores
#SBATCH --gres=gpu:1                  # Ask for 1 GPU
#SBATCH --mem=4G                    # Memory request; MB assumed if unit not specified
#SBATCH --time=02:00:00               # Time limit hrs:min:sec
#SBATCH --output=%x-%A_%a.out # fichier de sortie (%j = job ID)
#SBATCH --error=%x-%A_%a.err # fichier d’erreur (%j = job ID)
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
#SBATCH --array=0-2 # job array with 4 tasks (0 to 3)

module purge

source ../.venv/bin/activate

echo "visible nvidia gpus $(nvidia-smi)"
echo "Running job on $(hostname)"
echo "python: $(which python)"
echo "python-version $(python -V)"
echo "CUDA_DEVICE: $CUDA_VISIBLE_DEVICES"
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True #avoid fragmentation issues?

#python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
#python -c "import torch; print(f'cuda device: {torch.cuda.current_device()}')"

set -x # activer l’echo des commandes

DATASET=MSPKA_EMA_ita # MNGU0, mocha_timit, MSPKA_EMA_ita, pb2007

# Define speakers conditionally
if [ "$DATASET" = "MNGU0" ]; then
    SPEAKERS=(s1)
elif [ "$DATASET" = "mocha_timit" ]; then
    SPEAKERS=(faet0 ffes0 fsew0 maps0 mjjn0 msak0)
elif [ "$DATASET" = "MSPKA_EMA_ita" ]; then
    SPEAKERS=(cnz lls olm)
elif [ "$DATASET" = "pb2007" ]; then
    SPEAKERS=(spk1)
else
    echo "Unknown dataset: $DATASET"
    exit 1
fi

SPK=${SPEAKERS[$SLURM_ARRAY_TASK_ID]}

echo "Running inference for dataset: $DATASET, speaker: $SPK"

echo "computation start $(date)"
# launch your computation

echo "Running job on $(hostname)"
echo "python: $(which python)"
echo "python-version $(python -V)"

set -x # activer l’echo des commandes

echo "computation start $(date)"
# launch your computation

srun python -u ./arttts_inference.py  --data_dir /scratch2/ali/data \
                                    --save_dir /scratch2/ali/data/$DATASET/arttts/$SPK/arttts_pred/v1/grad_565 \
                                    --filelist_path resources/filelists/$DATASET/${SPK}_v1.txt \
                                    --version v1 \
                                    --ckpt_name grad_565.pt \
                                    --params_name params_v1 \
                                    --device cuda \
                                    --batch_size 16 \
                                    
echo "computation end : $(date)"