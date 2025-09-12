#!/bin/bash
#SBATCH --job-name=hifigan_infer     # Job name
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

DATASET=MNGU0 # MNGU0, mocha_timit, MSPKA_EMA_ita, pb2007
CKPT_NAME=grad_5000
MODEL_VERSION=v1
FILELIST_VERSION=v1
MAIN_DATA_DIR=/lustre/fsn1/projects/rech/rec/commun/data

for E in 200 1000 2000 3000 4000 5000; do
    CKPT_NAME=grad_${E}
    for MODEL_VERSION in v4; do
        if [ "$MODEL_VERSION" = "v1" ]; then
            FILELIST_VERSION=v1
        elif [ "$MODEL_VERSION" = "v1_1" ]; then
            FILELIST_VERSION=v1
        elif [ "$MODEL_VERSION" = "v2" ]; then
            FILELIST_VERSION=v2
        elif [ "$MODEL_VERSION" = "v3" ]; then
            FILELIST_VERSION=v1
        elif [ "$MODEL_VERSION" = "v4" ]; then
            FILELIST_VERSION=v4
        elif [ "$MODEL_VERSION" = "v5" ]; then
            FILELIST_VERSION=v1
        elif [ "$MODEL_VERSION" = "v5_preblock" ]; then
            FILELIST_VERSION=v1
        elif [ "$MODEL_VERSION" = "v2_phnmtext" ]; then
            FILELIST_VERSION=v2
        elif [ "$MODEL_VERSION" = "v4_phnmtext" ]; then
            FILELIST_VERSION=v4
        fi
        # Define speakers conditionally
        if [ "$DATASET" = "LJSpeech-1.1" ]; then
            #echo "Running hifigan inference for dataset: $DATASET, from SPARC encoded"
            #srun python -u ./hifigan_inference.py  --data_dir ${MAIN_DATA_DIR}/LJSpeech-1.1/encoded_audio_en/emasrc \
            #                                --save_dir ${MAIN_DATA_DIR}/LJSpeech-1.1/hifigan_pred/sparc \
            #                                --filelist_path resources/filelists/ljspeech/valid_${FILELIST_VERSION}.txt \
            #                                --main_dir ${MAIN_DATA_DIR} \
            #                                --device cuda \

            SRC_ART=decoder
            echo "Running hifigan inference for dataset: $DATASET, from : $SRC_ART"
            srun python -u ./hifigan_inference.py  --data_dir ${MAIN_DATA_DIR}/LJSpeech-1.1/arttts_pred/${MODEL_VERSION}/${CKPT_NAME} \
                                            --save_dir ${MAIN_DATA_DIR}/LJSpeech-1.1/hifigan_pred/${MODEL_VERSION}/${CKPT_NAME} \
                                            --filelist_path resources/filelists/ljspeech/valid_${FILELIST_VERSION}.txt \
                                            --main_dir ${MAIN_DATA_DIR} \
                                            --device cuda \
                                            --version ${MODEL_VERSION} \
                                            --src_art $SRC_ART
            SRC_ART=encoder
            echo "Running hifigan inference for dataset: $DATASET, from : $SRC_ART"
            srun python -u ./hifigan_inference.py  --data_dir ${MAIN_DATA_DIR}/LJSpeech-1.1/arttts_pred/${MODEL_VERSION}/${CKPT_NAME} \
                                            --save_dir ${MAIN_DATA_DIR}/LJSpeech-1.1/hifigan_pred/${MODEL_VERSION}/${CKPT_NAME} \
                                            --filelist_path resources/filelists/ljspeech/valid_${FILELIST_VERSION}.txt \
                                            --main_dir ${MAIN_DATA_DIR} \
                                            --device cuda \
                                            --version ${MODEL_VERSION} \
                                            --src_art $SRC_ART
        else
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

            #echo "Running hifigan inference for dataset: $DATASET, from SPARC encoded"
            #srun python -u ./hifigan_inference.py  --data_dir ${MAIN_DATA_DIR}/$DATASET/arttts/$SPK/encoded_audio_en/emasrc \
            #                                    --save_dir ${MAIN_DATA_DIR}/$DATASET/arttts/$SPK/hifigan_pred/sparc \
            #                                    --filelist_path resources/filelists/$DATASET/${SPK}_${FILELIST_VERSION}.txt \
            #                                    --main_dir ${MAIN_DATA_DIR} \
            #                                    --device cuda \

            SRC_ART=decoder
            echo "Running hifigan inference for dataset: $DATASET, speaker: $SPK, from : $SRC_ART"
            srun python -u ./hifigan_inference.py  --data_dir ${MAIN_DATA_DIR}/$DATASET/arttts/$SPK/arttts_pred/${MODEL_VERSION}/${CKPT_NAME} \
                                                --save_dir ${MAIN_DATA_DIR}/$DATASET/arttts/$SPK/hifigan_pred/${MODEL_VERSION}/${CKPT_NAME} \
                                                --filelist_path resources/filelists/$DATASET/${SPK}_${FILELIST_VERSION}.txt \
                                                --main_dir ${MAIN_DATA_DIR} \
                                                --device cuda \
                                                --version ${MODEL_VERSION} \
                                                --src_art $SRC_ART
            SRC_ART=encoder
            echo "Running hifigan inference for dataset: $DATASET, speaker: $SPK, from : $SRC_ART"
            srun python -u ./hifigan_inference.py  --data_dir ${MAIN_DATA_DIR}/$DATASET/arttts/$SPK/arttts_pred/${MODEL_VERSION}/${CKPT_NAME} \
                                                --save_dir ${MAIN_DATA_DIR}/$DATASET/arttts/$SPK/hifigan_pred/${MODEL_VERSION}/${CKPT_NAME} \
                                                --filelist_path resources/filelists/$DATASET/${SPK}_${FILELIST_VERSION}.txt \
                                                --main_dir ${MAIN_DATA_DIR} \
                                                --device cuda \
                                                --version ${MODEL_VERSION} \
                                                --src_art $SRC_ART
        fi
    done
done

echo "computation end : $(date)"