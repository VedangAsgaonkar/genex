## Running commands with hyperparameters:
Main training script
```
python -u exec/new_train_LSH.py -d disease_pred --num_classes 42 -gpu {gpu} -g_freq 2 --num_sets 20 --set_embedding_dim 20 --encoder_output_dim 21 --latent_dim 10 --num_heads 7 --num_layers 5  -sm fixed_length -bs 20 --budget 40 -c {c} -sf save_tensorized/ -smf save_metrics/ -lr 0.001 --epochs=100 --fix_ug --ug 0.2 --batchnorm
```
Script for training on generated data
```
python -u exec/new_train_LSH.py -d disease_pred --num_classes 42 -gpu {gpu} -g_freq 2 --num_sets 20 --set_embedding_dim 20 --encoder_output_dim 21 --latent_dim 10 --num_heads 7 --num_layers 5  -sm fixed_length -bs 20 --budget 40 -c {c} -sf save_tensorized/ -smf save_metrics/ -lr 0.001 --epochs=100 --fix_ug --ug 0.2 --batchnorm --choice_frac 0.1
```
Use appropriate flags as indicated in the main README