DATA_ROOT=../datasets

train_alg=dagger

features=clip.h14
ft_dim=1024
obj_features=vitbase
obj_ft_dim=768

ngpus=2
bs=8
seed=0

name=${train_alg}-${features}-envedit
name=${name}-seed.${seed}
name=${name}-aug.mp3d.prevalent.hm3d_gibson.envdrop.init.190k


outdir=${DATA_ROOT}/NavRAG/exprs_map/finetune/${name}-aug.hm3d.envdrop

flag="--root_dir ${DATA_ROOT}
      --dataset NavRAG
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert      

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy spl
      --train_alg ${train_alg}
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 30
      --max_instr_len 200

      --batch_size ${bs}
      --lr 1e-5
      --iters 40000
      --log_every 1000
      --aug_times 9

      --env_aug

      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --ml_weight 0.15

      --feat_dropout 0.4
      --dropout 0.5
      
      --gamma 0."


# train
CUDA_VISIBLE_DEVICES=$1 python3 -m torch.distributed.launch --master_port $2 --nproc_per_node=${ngpus} NavRAG/main_nav.py $flag  \
      --tokenizer bert \
      --bert_ckpt_file ../datasets/NavRAG/exprs_map/pretrain/cmt-clip.vit.h14-mlm.sap-init.lxmert/ckpts/model_step_200000.pt

# # test
# CUDA_VISIBLE_DEVICES='0' python NavRAG/main_nav.py $flag  \
#       --tokenizer bert \
#       --resume_file ../datasets/NavRAG/trained_models/finetune/duet_vit-h14_ft_best_val_unseen \
#       --test #--submit