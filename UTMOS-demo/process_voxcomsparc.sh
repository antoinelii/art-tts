#!/bin/bash
#SBATCH --job-name=predict_utmos     # Job name
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

source .venv/bin/activate

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

dataset=VoxCommunis
generator_version=sparc_multi
split=dev-1h

for version in msml1h; do
	CKPT_NAME=grad_${E}
	utmos_version_dir=${MAIN_DATA_DIR}/${dataset}/UTMOS_data/decoder_files_sparc/${split}/${generator_version}
	mkdir -p  ${utmos_version_dir}
	find ${MAIN_DATA_DIR}/${dataset}/${split} -type f -path "*/hifigan_pred/sparc/${generator_version}/*.wav" \
		-exec cp {} ${utmos_version_dir} \;
	uv run predict.py --mode predict_dir \
			--inp_dir ${utmos_version_dir} \
			--bs 32 \
			--out_path ${utmos_version_dir}/utmos.csv
done

echo "computation end : $(date)"
