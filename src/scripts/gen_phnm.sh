#!/bin/bash
#SBATCH --job-name=gen_phnms     # Job name
#SBATCH --partition=cpu_p1             # Take a node from the 'gpu' partition
#SBATCH --export=ALL                  # Export your environment to the compute node

#SBATCH --output=%x-%A_%a.out # fichier de sortie (%j = job ID)
#SBATCH --error=%x-%A_%a.err # fichier d’erreur (%j = job ID)
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=1 # reserver 4 taches (ou processus)
#SBATCH --ntasks-per-node=1 # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --cpus-per-task=1 # reserver 10 CPU par tache (et memoire associee)
#SBATCH --time=1:00:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
#SBATCH --account=rec@cpu
#SBATCH --array=0 # job array with 4 tasks (0 to 3)

module purge

source ../.venv/bin/activate

#mocha
#SPEAKERS=(faet0 ffes0 fsew0 maps0 mjjn0 msak0)
#SPK=${SPEAKERS[$SLURM_ARRAY_TASK_ID]}
#
#mngu0
SPEAKERS=(s1)
SPK=${SPEAKERS[$SLURM_ARRAY_TASK_ID]}
#
##mspka
#SPEAKERS=(cnz lls olm)
#SPK=${SPEAKERS[$SLURM_ARRAY_TASK_ID]}

#pb2007
#SPEAKERS=(spk1)
#SPK=${SPEAKERS[$SLURM_ARRAY_TASK_ID]}

echo "Running job on $(hostname)"
echo "python: $(which python)"
echo "python-version $(python -V)"

set -x # activer l’echo des commandes

echo "computation start $(date)"
# launch your computation

#srun python -u ./generate_phnm3.py --phnm_dir /lustre/fsn1/projects/rech/rec/commun/data/mocha_timit/src_data/$SPK \
#                                        --save_dir /lustre/fsn1/projects/rech/rec/commun/data/mocha_timit/arttts/$SPK/phnm3 \
#                                        --dataset_name mocha \
#
srun python -u ./generate_phnm3.py --phnm_dir /lustre/fsn1/projects/rech/rec/commun/data/MNGU0/src_data/$SPK/phone_labels \
                                        --save_dir /lustre/fsn1/projects/rech/rec/commun/data/MNGU0/arttts/$SPK/phnm3 \
                                        --dataset_name mngu0

#srun python -u ./generate_phnm3.py --phnm_dir /lustre/fsn1/projects/rech/rec/commun/data/MSPKA_EMA_ita/src_data/${SPK}_1.0.0/lab_1.0.0 \
#                                        --save_dir /lustre/fsn1/projects/rech/rec/commun/data/MSPKA_EMA_ita/arttts/$SPK/phnm3 \
#                                        --dataset_name mspka
#
#srun python -u ./generate_phnm3.py --phnm_dir /lustre/fsn1/projects/rech/rec/commun/data/pb2007/src_data/$SPK \
#                                        --save_dir /lustre/fsn1/projects/rech/rec/commun/data/pb2007/arttts/$SPK/phnm3 \
#                                        --dataset_name pb2007

echo "computation end : $(date)"
