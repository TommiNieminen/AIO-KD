#!/bin/bash

#SBATCH -J preprocess
#SBATCH -o preprocess.%j.out
#SBATCH -e preprocess.%j.err
#SBATCH -A project_2006944
#SBATCH -t 00:15:00
#SBATCH -p test
#SBATCH --cpus-per-task=16
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem-per-cpu=4G
echo "Starting at `date`"

python -m fairseq_cli.preprocess --source-lang en --target-lang fi \
    --testpref ./wmt17/test \
    --destdir ./data-bin/wmt17 \
    --workers 16 \
    --bpe sentencepiece \
    --srcdict ./data-bin/tc_Tatoeba-Challenge-v2023-09-26-tokenized.en-fi/dict.en.txt \
    --tgtdict ./data-bin/tc_Tatoeba-Challenge-v2023-09-26-tokenized.en-fi/dict.fi.txt

echo "Finishing at `date`"
