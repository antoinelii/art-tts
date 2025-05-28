#!/bin/bash
#SBATCH --job-name=encode_audio_en     # Job name
#SBATCH --partition=gpu_p2             # Take a node from the 'gpu' partition
# #SBATCH --export=ALL                  # Export your environment to the compute node

#SBATCH --output=%x-%j.out # fichier de sortie (%j = job ID)
#SBATCH --error=%x-%j.err # fichier d’erreur (%j = job ID)
# #SBATCH --constraint=v100-16g # demander des GPU a 16 Go de RAM
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=4 # reserver 4 taches (ou processus)
#SBATCH --ntasks-per-node=4 # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:4 # reserver 4 GPU
#SBATCH --cpus-per-task=3 # reserver 10 CPU par tache (et memoire associee)
#SBATCH --time=10:00:00 # temps maximal d’allocation "(HH:MM:SS)"
# #SBATCH --qos=qos_gpu-dev # QoS
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
#SBATCH --account=rec@v100 # comptabilite V100

module purge
#module load miniconda3/24.3.0
#conda init
#source "$(conda info --base)/etc/profile.d/conda.sh"
#conda create -n arttra python=3.11
#conda activate arttra
#echo "Activated conda environment: $(conda info --envs)"
#echo "Current environment: $(conda info --envs | grep '*' | cut -d' ' -f1)"
#echo "pip list: $(pip list)"
#cd src/speech-articulatory-coding
#pip install --upgrade pip setuptools wheel
#pip install -e .
#cd ../..
#echo "pwd: $(pwd)"
#echo "pip list: $(pip list)"
#echo "pip list: $(pip list)"

source .venv/bin/activate

echo "visible nvidia gpus $(nvidia-smi)"
echo "Running job on $(hostname)"
echo "python: $(which python)"
echo "python-version $(python -V)"
echo "CUDA_DEVICE: $CUDA_VISIBLE_DEVICES"

#python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
#python -c "import torch; print(f'cuda device: {torch.cuda.current_device()}')"

set -x # activer l’echo des commandes

echo "computation start $(date)"
# launch your computation

srun python -u scripts/encode_audio.py --device cuda \
                                    --wav_dir ../../data/LJSpeech-1.1/wavs \
                                    --save_dir ../../data/LJSpeech-1.1/encoded_audio_en \
                                    --ckpt_path ckpt/sparc_en.ckpt

echo "computation end : $(date)"
