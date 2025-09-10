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

DATASET=VoxCommunis
MAIN_DATA_DIR=/lustre/fsn1/projects/rech/rec/commun/data
SPLIT=test-1h
for E in 1000 2000 3000 4000; do
    CKPT_NAME=grad_${E}
    for MODEL_VERSION in msml1h; do
        #LANGs that were present in the training data
        #for LANG in  ja ba ro hi uz tt el sr mt yo be uk hy-AM sk ckb tr vi bg ta sv-SE id tk kmr dv zh-HK bn mn zh-CN yue lij fr hsb cv nl ug mr it lt sl pa-IN ru cs ml nan-tw th pt ky pl ca hu rw; do
        #for LANG in  ja ba ro hi uz tt el sr mt yo be uk hy-AM sk ckb tr vi bg ta sv-SE id tk kmr dv zh-HK bn mn zh-CN yue lij fr hsb cv nl ug mr it lt sl pa-IN ru cs ml nan-tw th pt ky pl ca hu rw; do
        #zero shot LANGS
        for LANG in eu ka ab gn sw ha ko myv; do
            echo "Computing PCCs for dataset: $DATASET, src_art : $MODEL_VERSION $CKPT_NAME decoder"
            srun python -u ./quanti_art_voxcom.py  --data_dir ${MAIN_DATA_DIR}/${DATASET}/${SPLIT}/arttts_pred/${MODEL_VERSION}/${CKPT_NAME} \
                                                --save_dir ${MAIN_DATA_DIR}/${DATASET}/${SPLIT}/analysis/ \
                                                --sparc_dir ${MAIN_DATA_DIR}/${DATASET}/encoded_audio_multi/${LANG} \
                                                --manifest_path ${MAIN_DATA_DIR}/${DATASET}/${SPLIT}/manifests/${LANG}.tsv \
                                                --version ${MODEL_VERSION} \
                                                --params_name params_${MODEL_VERSION} \
                                                --ckpt_name ${CKPT_NAME}
        done        
    done
done
echo "computation end : $(date)"