cd ..

python train_cpn_ova.py  \
    --arch  resnet18  \
    --train_dir '/data/datasets/imagenet/train'  \
    --test_dir '/data/datasets/imagenet/val'  \
    --thres_type 'multi'  \
    --mix_ova 0.80  \
    --pl_normalized  \
    --temp 1.5  \
    --model-dir 'checkpoint/'  \
    --num_class 10
