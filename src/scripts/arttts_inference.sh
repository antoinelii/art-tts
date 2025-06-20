#!/bin/bash
#SBATCH --job-name=arttts_infer     # Job name
#SBATCH --partition=gpu_p2             # Take a node from the 'gpu' partition
# #SBATCH --export=ALL                  # Export your environment to the compute node

#SBATCH --output=%x-%j.out # fichier de sortie (%j = job ID)
#SBATCH --error=%x-%j.err # fichier d’erreur (%j = job ID)
# #SBATCH --constraint=v100-16g # demander des GPU a 16 Go de RAM
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=1 # reserver 4 taches (ou processus)
#SBATCH --ntasks-per-node=1 # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1 # reserver 4 GPU
#SBATCH --cpus-per-task=3 # reserver 10 CPU par tache (et memoire associee)
#SBATCH --time=2:00:00 # temps maximal d’allocation "(HH:MM:SS)"
# #SBATCH --qos=qos_gpu-dev # QoS
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
#SBATCH --account=rec@v100 # comptabilite V100


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

srun python -u arttts_inference.py  --data_dir ../../data/ \
                                    --save_dir ../../data/LJSpeech-1.1/arttts_pred/v1/grad_565 \
                                    --filelist_path resources/filelists/ljspeech/valid_v1.txt \
                                    --version v1 \
                                    --ckpt_name grad_565.pt \
                                    --params_name params_v1 \
                                    --device cuda \
                                    --batch_size 16 \

echo "computation end : $(date)"