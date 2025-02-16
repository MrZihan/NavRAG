
NODE_RANK=0
NUM_GPUS=2
outdir=../datasets/NavRAG/exprs_map/pretrain/cmt-clip.vit.h14-mlm.sap-init.lxmert

# train
CUDA_VISIBLE_DEVICES=$1 python -m torch.distributed.launch \
    --master_port $2 \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK \
    train_rag.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config config/rag_model_config_clip-h14.json \
    --config config/rag_pretrain_hm3d+mp3d+gibson_clip-h14.json \
    --output_dir $outdir 
