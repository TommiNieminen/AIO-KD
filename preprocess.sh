#!/bin/bash

#SBATCH -J preprocess
#SBATCH -o preprocess.%j.out
#SBATCH -e preprocess.%j.err
#SBATCH -A project_2006944
#SBATCH -t 24:00:00
#SBATCH -p small
#SBATCH --cpus-per-task=16
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem-per-cpu=4G
echo "Starting at `date`"

python -m fairseq_cli.preprocess --source-lang en --target-lang fi \
    --trainpref ./tc_Tatoeba-Challenge-v2023-09-26/train \
    --validpref ./tc_Tatoeba-Challenge-v2023-09-26/valid \
    --testpref ./tc_Tatoeba-Challenge-v2023-09-26/test \
    --destdir ./data-bin/tc_Tatoeba-Challenge-v2023-09-26-tokenized.en-fi \
    --workers 16 \
    --joined-dictionary \
    --bpe sentencepiece
    

echo "Finishing at `date`"
