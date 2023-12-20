## Running commands with hyperparameters:
Main training script
```
python -u exec/new_train_tiny_imagenet.py -d cifar100 -ns 5 -sm fixed_length -c {c} -nc 100 --epochs=80 -bs 129 -lr 0.0001 -g_freq 2 --budget 40 --fix_ug --ug 0.8 --batchnorm -gpu {gpu} -sf save_tensorized/ -smf save_metrics/
```
Script for training on generated data
```
python -u exec/new_train_LSH_choice_gen.py -d cifar100 -ns 5 -sm fixed_length -c {c} -nc 100 --epochs=80 -bs 129 -lr 0.0001 -g_freq 2 --budget 40 --fix_ug --ug 0.8 --batchnorm -gpu {gpu} -sf save_tensorized/ -smf save_metrics/ --choice_frac 0.1
```
Use appropriate flags as indicated in the main README