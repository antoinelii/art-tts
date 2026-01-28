# On your local machine:
ssh ure35aq@jean-zay "cd /lustre/fsn1/projects/rech/rec/commun/data/VoxCommunis && find . -type f -path '*/UTMOS_data/***.csv'" \
  > utmos_csv_files.txt

rsync -azuP --files-from=utmos_csv_files.txt --relative ure35aq@jean-zay:/lustre/fsn1/projects/rech/rec/commun/data/VoxCommunis/ data/VoxCommunis/

