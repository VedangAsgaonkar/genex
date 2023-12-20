## Running commands with hyperparameters:
Main training script
```
python -u exec/new_train_LSH.py -d mnist --num_classes 10 -gpu {gpu} -g_freq 2 --num_sets 10 -sm fixed_length -bs 1024 --budget 25 -c {c} -sf save_tensorized/ -lr 0.001 --epochs=100 --fix_ug --ug 0.2 --greedy_threshold 0.2 --batchnorm
```
Script for training on generated data
```
python -u exec/new_train_LSH_choice_gen.py -d mnist --num_classes 10 -gpu {gpu} -g_freq 2 --num_sets 10 -sm fixed_length -bs 1024 --budget 25 -c {c} -sf save_tensorized/ -lr 0.001 --epochs=100 --fix_ug --ug 0.2 --greedy_threshold 0.2 --batchnorm --choice_frac 0.1
```
Use appropriate flags as indicated in the main README