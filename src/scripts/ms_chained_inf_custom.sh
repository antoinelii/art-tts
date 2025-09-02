#!/bin/sh
JID_JOB1=`sbatch scripts/ms_arttts_inf_custom.sh | cut -d " " -f 4`
JID_JOB2=`sbatch --dependency=afterok:$JID_JOB1 --kill-on-invalid-dep=yes scripts/ms_quanti_comp_custom.sh | cut -d " " -f 4`
JID_JOB3=`sbatch --dependency=afterok:$JID_JOB1 --kill-on-invalid-dep=yes scripts/ms_hifigan_inf_custom.sh | cut -d " " -f 4`