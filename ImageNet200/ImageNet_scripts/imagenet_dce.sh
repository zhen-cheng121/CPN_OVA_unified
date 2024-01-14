cd ..

python train_cpn_dce.py  \
    --arch  resnet18  \
    --train_dir '/data/datasets/imagenet/train'  \
    --test_dir '/data/datasets/imagenet/val'  \
    --temp 1.5  \
    --model-dir 'checkpoint/'  \
    --num_class 10
