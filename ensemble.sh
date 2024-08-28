#!/bin/bash
#SBATCH -J evaluate
#SBATCH -o evaluate.%j.out
#SBATCH -e evaluate.%j.err
#SBATCH -A project_2007095
#SBATCH -t 24:00:00
#SBATCH -p small
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=128G
echo "Starting at `date`"

python3 fairseq_cli/generate.py ./data-bin/tatoeba-test-v2021-08-07.eng-fin \
--path ./ckpts/tc_Tatoeba-Challenge-v2023-09-26.en-fi.transformer/stage2/checkpoint_best.pt \
   --encoder-layer-to-infer 6 --decoder-layer-to-infer 6  \
--beam 5 --remove-bpe=sentencepiece --cpu > ./eval_10012024/tc2021_res-ensemble.out
 # calculate BLEU score   
 #bash scripts/compound_split_bleu.sh 

echo "Finishing at `date`"
