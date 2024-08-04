# export NCCL_P2P_DISABLE=1
# export CUDA_VISIBLE_DEVICES=4,5,6,7


python3.9   -m torch.distributed.launch --nproc_per_node=4  --use_env main_tempWarm.py --model deit_small_patch16_224 \
 --batch-size 2048  --data-path <imagenet>  --output_dir <output_dir> \
 --epochs 30  --bce-loss --data-set IMNET-500 --ova_loss 1 --regCE 0.95 --cpn_target_threshold 0.0 \
 --mixup 0 --cutmix 0.0  \
 --lr 5e-4  --cpn_temp 1.00   --cpn_Mclass  --cpn_thres_type  one   --thres_init --thres_init --temp_warmup_S1 --thres_init_mul 6
