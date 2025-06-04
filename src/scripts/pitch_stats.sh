#!/bin/bash
#SBATCH --job-name=pitch_stats     # Job name
#SBATCH --partition=cpu_p1             # Take a node from the 'gpu' partition
# #SBATCH --export=ALL                  # Export your environment to the compute node

#SBATCH --output=%x-%A_%a.out # fichier de sortie (%j = job ID)
#SBATCH --error=%x-%A_%a.err # fichier d’erreur (%j = job ID)
# #SBATCH --constraint=v100-16g # demander des GPU a 16 Go de RAM
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=1 # reserver 4 taches (ou processus)
#SBATCH --ntasks-per-node=1 # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --time=0:10:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
#SBATCH --account=rec@cpu # comptabilite V100

module purge

source ../.venv/bin/activate

echo "Running job on $(hostname)"
echo "python: $(which python)"
echo "python-version $(python -V)"

set -x # activer l’echo des commandes

echo "computation start $(date)"
# launch your computation

srun python -u ./pitch_stats.py

echo "computation end : $(date)"
