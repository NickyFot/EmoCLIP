#!/bin/bash
#$ -l h_rt=3:0:0
#$ -l h_vmem=7.5G
#$ -pe smp 16
#$ -l gpu=2
#$ -l gpu_type=ampere
#$ -cwd
#$ -j y
#$ -m be

module load miniconda/4.12.0
conda activate venv

#model parameters
model_basename=zeroshot
log_dir='logs/'
batch_size=128
dataset_root="/datasets/DFEW"
dataset_name=DFEW
downsample=4
clip_len=32
pretrained='logs/checkpoints/2023-03-08/models/4.pth'

exp_name=$(date "+%Y-%m-%d")
exp_name="${exp_name}_${model_basename}_${dataset_name}"
#torch.distributed parameters
num_gpus=2

for i in {1..5}
do
  python -m torch.distributed.launch --nproc_per_node=$num_gpus --nnodes=1 --node_rank 0 eval.py \
      --fold $i \
      --log_dir $log_dir \
      --batch_size $batch_size \
      --dataset_root $dataset_root \
      --dataset_name $dataset_name \
      --model_basename $model_basename\
      --num_epochs $num_epochs \
      --optim $optim \
      --lr $lr \
      --downsample $downsample \
      --exp_name $exp_name \
      --clip_len $clip_len \
      --pretrained $pretrained
done
