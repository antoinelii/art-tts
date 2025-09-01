#!/bin/bash
#SBATCH --job-name=arttts_infer     # Job name
#SBATCH --partition=gpu_p2             # Take a node from the 'gpu' partition
# #SBATCH --export=ALL                  # Export your environment to the compute node

#SBATCH --output=%x-%A_%a.out # fichier de sortie (%j = job ID)
#SBATCH --error=%x-%A_%a.err # fichier d’erreur (%j = job ID)
# #SBATCH --constraint=v100-16g # demander des GPU a 16 Go de RAM
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=1 # reserver 4 taches (ou processus)
#SBATCH --ntasks-per-node=1 # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1 # reserver 4 GPU
#SBATCH --cpus-per-task=3 # reserver 10 CPU par tache (et memoire associee)
#SBATCH --time=4:00:00 # temps maximal d’allocation "(HH:MM:SS)"
# #SBATCH --qos=qos_gpu-dev # QoS
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
#SBATCH --account=rec@v100 # comptabilite V100
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

DATASET=GradTTS_samples
CKPT_NAME=grad_10000
MODEL_VERSION=v2_full
FILELIST_PATH=resources/filelists/gradtts_samples/gradtts_samples.txt
MAIN_DATA_DIR=/lustre/fsn1/projects/rech/rec/commun/data

echo "Running mel inference for dataset: $DATASET"
srun python -u arttts_inference.py  --data_dir ${MAIN_DATA_DIR}/ \
                                --save_dir ${MAIN_DATA_DIR}/${DATASET}/arttts_pred/${MODEL_VERSION}/${CKPT_NAME} \
                                --filelist_path $FILELIST_PATH \
                                --version ${MODEL_VERSION} \
                                --ckpt_name ${CKPT_NAME}.pt \
                                --params_name params_v2 \
                                --device cuda \
                                --batch_size 1 \
                                --use_align 0 \
                                --max_samples 0

SRC_ART=decoder
echo "Running vocoder inference for dataset: $DATASET, speaker: $SPK, from : $SRC_ART"
srun python -u ./vocoder_inference.py  --data_dir ${MAIN_DATA_DIR}/${DATASET}/arttts_pred/${MODEL_VERSION}/${CKPT_NAME} \
                                    --save_dir ${MAIN_DATA_DIR}/${DATASET}/hifigan_pred/${MODEL_VERSION}/${CKPT_NAME} \
                                    --filelist_path $FILELIST_PATH \
                                    --main_dir ${MAIN_DATA_DIR} \
                                    --device cuda \
                                    --version ${MODEL_VERSION} \
                                    --src_mel $SRC_ART

echo "computation end : $(date)"