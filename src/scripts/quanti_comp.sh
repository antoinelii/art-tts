#!/bin/bash
#SBATCH --job-name=quanti_comp     # Job name
#SBATCH --partition=cpu_p1             # Take a node from the 'gpu' partition
# #SBATCH --export=ALL                  # Export your environment to the compute node

#SBATCH --output=%x-%A_%a.out # fichier de sortie (%j = job ID)
#SBATCH --error=%x-%A_%a.err # fichier d’erreur (%j = job ID)
#SBATCH --ntasks=1 # reserver 4 taches (ou processus)
#SBATCH --cpus-per-task=3 # reserver 10 CPU par tache (et memoire associee)
#SBATCH --time=1:00:00 # temps maximal d’allocation "(HH:MM:SS)"
# #SBATCH --qos=qos_cpu-dev # QoS
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
#SBATCH --account=rec@cpu # comptabilite V100
#SBATCH --array=0 # job array with 4 tasks (0 to 3)


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

echo "computation start $(date)"
# launch your computation

MAIN_DATA_DIR=/lustre/fsn1/projects/rech/rec/commun/data

for DATASET in MNGU0 MSPKA_EMA_ita mocha_timit; do
    for MODEL_VERSION in v2_full; do
        for E in 2000 4000 6000 8000 10000 ; do
            CKPT_NAME=grad_${E}
            echo "Computing PCCs for dataset: $DATASET, src_art : $MODEL_VERSION $CKPT_NAME decoder"
            srun python -u ./quanti_art_comp.py  --main_data_dir ${MAIN_DATA_DIR} \
                                                --dataset ${DATASET} \
                                                --version ${MODEL_VERSION} \
                                                --ckpt_name ${CKPT_NAME}
        done
    done
done
echo "computation end : $(date)"
