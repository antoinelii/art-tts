#rsync -azuP  jean-zay:/lustre/fsn1/projects/rech/rec/commun/data/VoxCommunis/pb2007/analysis/ data/VoxCommunis/pb2007/analysis/
#rsync -azuP  jean-zay:/lustre/fsn1/projects/rech/rec/commun/data/VoxCommunis/MNGU0/analysis/ data/VoxCommunis/MNGU0/analysis/
#rsync -azuP  jean-zay:/lustre/fsn1/projects/rech/rec/commun/data/VoxCommunis/MSPKA_EMA_ita/analysis/ data/VoxCommunis/MSPKA_EMA_ita/analysis/
#rsync -azuP  jean-zay:/lustre/fsn1/projects/rech/rec/commun/data/VoxCommunis/mocha_timit/analysis/ data/VoxCommunis/mocha_timit/analysis/
#rsync -azuP  jean-zay:/lustre/fsn1/projects/rech/rec/commun/data/VoxCommunis/test-20h/analysis/ data/VoxCommunis/test-20h/analysis/
#rsync -azuP  jean-zay:/lustre/fsn1/projects/rech/rec/commun/data/VoxCommunis/dev-1h/analysis/ data/VoxCommunis/dev-1h/analysis/

# On your local machine:
ssh ure35aq@jean-zay "cd /lustre/fsn1/projects/rech/rec/commun/data/VoxCommunis && find . -type f -path '*/hifigan_pred/***.wav'" \
  > vc_hifipreds.txt

rsync -azuP --files-from=vc_hifipreds.txt --relative ure35aq@jean-zay:/lustre/fsn1/projects/rech/rec/commun/data/VoxCommunis/ data/VoxCommunis/

