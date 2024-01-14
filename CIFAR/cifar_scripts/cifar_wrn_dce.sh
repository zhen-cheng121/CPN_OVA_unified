# scripts for training on CIFAR-10/CIFAR-100
cd ..

python train_cifar_dce.py \
    --model wrn-28-10 \
    --model-dir './checkpoint/cifar10/wrn-28-10_pl0p05_temper2p0/'  \
    --dataset cifar-10  \
    --data-dir 'datasets/'  \
    --pl_weight 0.05 \
    --temp 2.00

# python train_cifar_dce.py \
#     --model wrn-28-10 \
#     --model-dir './checkpoint/cifar100/wrn-28-10_pl0p05_temper2p0/'  \
#     --dataset cifar-100  \
#     --data-dir 'datasets/'  \
#     --pl_weight 0.05 \
#     --temp 2.00
