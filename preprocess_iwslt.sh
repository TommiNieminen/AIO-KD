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

python -m fairseq_cli.preprocess --source-lang de --target-lang en \
    --testpref ./iwslt14.tokenized.de-en/test \
    --destdir ./data-bin/iwslt14.tokenized.de-en \
    --workers 16 \
    --srcdict ./data-bin/iwslt14.tokenized.de-en/dict.de.txt \
    --tgtdict ./data-bin/iwslt14.tokenized.de-en/dict.en.txt 

echo "Finishing at `date`"
