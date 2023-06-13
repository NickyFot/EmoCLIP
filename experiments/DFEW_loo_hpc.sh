#!/bin/bash
#$ -l h_rt=48:0:0
#$ -l h_vmem=7.5G
#$ -pe smp 24
#$ -l gpu=3
#$ -l gpu_type=ampere
#$ -cwd
#$ -j y
#$ -m be

module load miniconda/4.12.0
conda activate venv

#model parameters
model_basename=loo
log_dir='logs/'
batch_size=128
num_epochs=30
lr=0.001
dataset_root="/datasets/DFEW"
dataset_name=DFEW
downsample=4
clip_len=32
optim="SGD"

#exp_name=$(date "+%Y-%m-%d")
exp_name="2023-05-05"
exp_name="${exp_name}_${model_basename}_${dataset_name}"
#torch.distributed parameters
num_gpus=3

for j in {1..6}
do
  for i in {1..5}
  do
    python -m torch.distributed.launch --nproc_per_node=$num_gpus --nnodes=1 --node_rank 0 train_loo.py \
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
        --emo $j

  done
done
