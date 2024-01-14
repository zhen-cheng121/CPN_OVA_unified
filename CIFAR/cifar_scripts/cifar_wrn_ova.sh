# scripts for training on CIFAR-10/CIFAR-100
cd ..

python train_cifar_ova.py \
    --model wrn-28-10 \
    --model-dir './checkpoint/cifar10/wrn-28-10_thresMULTI_pl0p05norm_temper3p0/'  \
    --dataset cifar-10  \
    --data-dir 'datasets/'  \
    --pl_weight 0.05 \
    --temp 3.00 \
    --thres_type 'multi'  \
    --pl_normalized  \
    --alpha_ova 0.98


# python train_cifar_ova.py \
#     --model wrn-28-10 \
#     --model-dir './checkpoint/cifar100/wrn-28-10_thresMULTI_pl0p05norm_temper3p0/'  \
#     --dataset cifar-100  \
#     --data-dir 'datasets/'  \
#     --pl_weight 0.10 \
#     --temp 5.00 \
#     --thres_type 'multi'  \
#     --pl_normalized  \
#     --alpha_ova 0.95

