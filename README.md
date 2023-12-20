## Running Instructions
The overall procedure for training and testing GenEx is:

* Use ```sample_initial_sets.py``` to generate the sets of observed features $\mathcal{O}_i$
* Use ```cluster_LSH.py``` to produce the buckets
* Train the generator using the ```--train_gen``` flag on the main training script for the dataset
* Then for each bucket make use of ```--warm_start``` for warm starting the classifier, ```--warm_end``` for the final training of the classifier. This must be done on both the main training script and the scipt for training the classifier with partially generated data (called ```new_train_LSH_choice_gen.py```). Greedy must be done only on the main training script, for ```new_train_LSH_choice_gen.py```, use the ```--no_greedy``` flag.
* Test on ```new_train_LSH_choice_gen.py``` using the ```--test``` flag. The ```--warm_end_load_T``` flag can be used to specify the budget $q_{\max}$ that we are using (specified as absolute number of features).

Detailed running instructions for each dataset, with hyperparameters are documented in the respective directories.

## Requirements
```
Python 3.8.10
torch==1.13.1
torchvision==0.14.1
numpy==1.24.2
scikit-learn==1.2.1
efficientnet-pytorch==0.7.1
```