#!/bin/sh
JID_JOB1=`sbatch scripts/msml_arttts_inf.sh | cut -d " " -f 4`
JID_JOB2=`sbatch --dependency=afterok:$JID_JOB1 --kill-on-invalid-dep=yes scripts/msml_quanti_comp.sh | cut -d " " -f 4`
JID_JOB3=`sbatch --dependency=afterok:$JID_JOB1 --kill-on-invalid-dep=yes scripts/msml_hifigan_inf.sh | cut -d " " -f 4`