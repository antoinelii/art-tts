#!/bin/bash
#SBATCH --job-name=train_msml1h     # Job name
#SBATCH --partition=gpu_p6             # Take a node from the 'gpu' partition
# #SBATCH --export=ALL                  # Export your environment to the compute node
#SBATCH -C h100
#SBATCH --output=%x-%j.out # fichier de sortie (%j = job ID)
#SBATCH --error=%x-%j.err # fichier d’erreur (%j = job ID)
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=1 # reserver 4 taches (ou processus)
#SBATCH --ntasks-per-node=1 # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1 # reserver 4 GPU
#SBATCH --cpus-per-task=24 # reserver 10 CPU par tache (et memoire associee)
#SBATCH --time=100:00:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --qos=qos_gpu_h100-t4
# #SBATCH --time=2:00:00 # temps maximal d’allocation "(HH:MM:SS)"
# #SBATCH --qos=qos_gpu_h100-dev 
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
#SBATCH --account=rec@h100 # comptabilite V100

module purge
module load arch/h100
module load cuda/12.6.3

source ../.venv/bin/activate

echo "visible nvidia gpus $(nvidia-smi)"
echo "Running job on $(hostname)"
echo "python: $(which python)"
echo "python-version $(python -V)"
echo "CUDA_DEVICE: $CUDA_VISIBLE_DEVICES"

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'cuda device: {torch.cuda.current_device()}')"
python -c "import torch; print(f'device_count: {torch.cuda.device_count()}')"

set -x # activer l’echo des commandes

echo "computation start $(date)"
# launch your computation

export CUDA_LAUNCH_BLOCKING=1
echo "CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"

srun python -u ./train_msml1h.py

echo "computation end : $(date)"
