#!/bin/bash

#SBATCH -J 2train
#SBATCH -o 2train.%j.out
#SBATCH -e 2train.%j.err
#SBATCH -A project_2007095
#SBATCH -t 48:00:00
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
echo "Starting at `date`"

sample_student=2
encoder_layer_max_idx=6
encoder_layer_min_idx=2
decoder_layer_max_idx=6
decoder_layer_min_idx=2
kd_weight=5.5
ce_weight=1.
sml_weight=0.5
threshold=1.1

python -m fairseq_cli.train /scratch/project_2006944/tommi/aio_kd/AIO-KD/data-bin/iwslt14.tokenized.de-en \
  --task translation --arch transformer_iwslt_de_en --share-all-embeddings \
  --optimizer adam --lr 0.0005 -s de -t en --label-smoothing 0.1 \
  --max-tokens 4096 --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
  --criterion cross_entropy_with_subnetwork_distillation --seed 64 \
  --encoder-layer-max-idx $encoder_layer_max_idx --encoder-layer-min-idx $encoder_layer_min_idx --n-encoder-layer $encoder_layer_max_idx \
  --decoder-layer-max-idx $decoder_layer_max_idx --decoder-layer-min-idx $decoder_layer_min_idx  --n-decoder-layer $decoder_layer_max_idx \
  --mutual-weight $sml_weight --kd-weight $kd_weight --ce-weight $ce_weight  --sample-student-number $sample_student  \
  --no-epoch-checkpoints --detach-threshold $threshold --uniform-sample \
  --max-update 300000 --student-mutual-learning no_weight \
  --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --adam-betas '(0.9,0.98)' --save-dir ./ckpts/iwslt14_de_en/stage2 \
  --no-epoch-checkpoints --fp16 --dropout 0.3 \
  --finetune-from-model ./ckpts/iwslt14_de_en/stage1/checkpoint_best.pt
echo "Finishing at `date`"
