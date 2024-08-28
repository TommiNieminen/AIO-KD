#!/bin/bash
#SBATCH -J evaluate
#SBATCH -o evaluate.%j.out
#SBATCH -e evaluate.%j.err
#SBATCH -A project_2006944
#SBATCH -t 02:00:00
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
echo "Starting at `date`"

module load pytorch
source ../aio_env/bin/activate

for el in 2 3 4 5 6
   do for dl in 2 3 4 5 6
        do
         echo "Decoding: ""encoder layer:"$el", decoder layer:"$dl
         # generate translations
         fairseq-generate ./data-bin/tc_Tatoeba-Challenge-v2023-09-26-tokenized.en-fi \
	   --path ./ckpts/tc_Tatoeba-Challenge-v2023-09-26.en-fi.transformer/stage2/checkpoint_last.pt \
           --encoder-layer-to-infer $el --decoder-layer-to-infer $dl  \
	   --beam 5 --remove-bpe=sentencepiece > ./eval_tc/tc_res-e${el}d${dl}.out
         # calculate BLEU score   
         bash scripts/compound_split_bleu.sh ./eval_tc/tc_res-e${el}d${dl}.out
   done
done

echo "Finishing at `date`"
