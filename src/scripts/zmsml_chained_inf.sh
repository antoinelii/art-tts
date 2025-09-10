#!/bin/sh
JID_JOB1=`sbatch scripts/zmsml_arttts_inf.sh | cut -d " " -f 4`
JID_JOB2=`sbatch --dependency=afterok:$JID_JOB1 --kill-on-invalid-dep=yes scripts/zmsml_quanti_comp.sh | cut -d " " -f 4`
JID_JOB3=`sbatch --dependency=afterok:$JID_JOB1 --kill-on-invalid-dep=yes scripts/zmsml_hifigan_inf.sh | cut -d " " -f 4`