#!/bin/bash
#$ -l h_rt=30:0:0
#$ -l h_vmem=7G
#$ -pe smp 16
#$ -l gpu=2
#$ -l gpu_type=ampere
#$ -cwd
#$ -j y
#$ -m be

module load miniconda/4.12.0
conda activate venv

#model parameters
model_basename=supervised
log_dir='logs/'
batch_size=32
num_epochs=30
lr=0.001
dataset_root="datasets/MAFW"
dataset_name=MAFW
downsample=4
clip_len=32
optim="SGD"

exp_name="2023-04-12"
exp_name="${exp_name}_${model_basename}_${dataset_name}"
#torch.distributed parameters
num_gpus=2

for i in {3..5}
do
    #echo $i
    WANDB__SERVICE_WAIT=300 python -m torch.distributed.launch --nproc_per_node=$num_gpus --nnodes=1 --node_rank 0 \
    train_downstream.py \
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
    --fold $i \
    --clip_len $clip_len \
    --pretrained $pretrained
done
