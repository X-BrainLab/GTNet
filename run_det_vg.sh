#!/usr/bin/env bash

NEPOCH=50
LAMBDA1=10
COSEM_WEIGHT=0.1
RECONS_WEIGHT=0.1
LR=0.0001
BATCH_SIZE=1024
SYN_NUM=1200
RESSZ=2048
NDH=4096
NGH=4096
ATTSZ=300
NBFL=26
NZ=300
IOU=300
TFN=9
FIN=1
FON=3
EFTSG=1
AGF=12


## GZSL-OD
#python2 clswgan_action.py --nclass_all 51 --dataroot data_action --gzsl_od --manualSeed 806 --ngh $NGH --ndh $NDH --lambda1 $LAMBDA1 --critic_iter 5  --lr $LR \
#--cosem_weight $COSEM_WEIGHT --recons_weight $RECONS_WEIGHT --syn_num 50 --preprocessing --cuda --batch_size $BATCH_SIZE --nz $NZ --attSize $ATTSZ --resSize $RESSZ \
#--action_embedding i3d --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch $NEPOCH --dataset hmdb51 --split 1 --no_bg_file $NBFL

## GZSL
#python2 clswgan_action.py --nclass_all 51 --dataroot data_action --gzsl --manualSeed 806 --ngh $NGH --ndh $NDH --lambda1 $LAMBDA1 --critic_iter 5  --lr $LR \
#--cosem_weight $COSEM_WEIGHT --recons_weight $RECONS_WEIGHT --syn_num $SYN_NUM --preprocessing --cuda --batch_size $BATCH_SIZE --nz $NZ --attSize $ATTSZ --resSize $RESSZ \
#--action_embedding i3d --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch $NEPOCH --dataset hmdb51 --split 1

# ZSL
CUDA_VISIBLE_DEVICES=2 python2 IoUGAN_train.py --nclass_all 51 --dataroot data_action --manualSeed 806 --ngh $NGH --ndh $NDH --lambda1 $LAMBDA1 --critic_iter 5  --lr $LR \
--cosem_weight $COSEM_WEIGHT --recons_weight $RECONS_WEIGHT --syn_num $SYN_NUM --preprocessing --cuda --batch_size $BATCH_SIZE --nz $NZ --attSize $ATTSZ --resSize $RESSZ \
--action_embedding i3d --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch $NEPOCH --dataset vg --split 1 --no_bg_file $NBFL --save_name baseline \
--iou_information_size $IOU --file_number $FIN --folder_number $FON --second_gan $EFTSG --all_gt_file $AGF --total_file_number $TFN --bg_generate seen --sv vg_seen0.2

