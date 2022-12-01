# FedTree Challenge for ECNU


## Introduction

This is the Federated Learning Challenge for ECNU. The challenge is based on the [FedTree](https://github.com/Xtra-Computing/FedTree) framework, which is a federated learning framework for decision tree.

In this challenge, we will investigate federated learning for fraud detection. Fraud detection is a very important task. 
The United Nations estimated that the amount of money laundered globally in one year is 2 - 5% of global GDP. 
In reality, transaction data is distributed in multiple organizations (e.g., multiple banks). Thus, federated learning for fraud detection is a promising direction.

In this challenge, we will give you a dataset for fraud detection. You need to partition the dataset to multiple subsets to simulate the federated setting.
Then, you need to develop the training model based on FedTree. The goal is to achieve high AUPRC for fraud detection.


## Setup

### Step 1 - Install FedTree
Please refer to the [FedTree](https://github.com/Xtra-Computing/FedTree) repo to install the FedTree framework from source.

### Step 2 - Simulate the Federated Setting

You need to download the fraud detection dataset [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and put `creditcard.csv` under `data` directory. 
Then, you can run ```python train_test_split.py``` to split the dataset into training dataset and test dataset.


Then, you can create the partitions of the dataset to simulate the federated setting with the help of ``partitions/partition.py``, which has the following parameters

| Parameter   | Description                                                                                                                                                                           |
|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `n_parties` | Number of parties, default = `2`.                                                                                                                                                     |
| `partition` | The partition way. Options: `homo`, `noniid-labeldir`, `noniid-#label1` (or 2, 3, ..., which means the fixed number of labels each party owns), `iid-diff-quantity`. Default = `homo` |
| `init_seed` | The initial seed, default = `0`.                                                                                                                                                      |
| `datadir`   | The path of the dataset, default = `./data/creditcard_train.csv`.                                                                                                                     |
| `outputdir` | The path of the output directory, default = `./data/partitioned_creditcard/`.                                                                                                         |
| `beta`      | The concentration parameter of the Dirichlet distribution for heterogeneous partition, default = `0.5`.                                                                               |


In this challenge, you need to try the following six federated settings.

* n_parties = 2, partition = `homo`
* n_parties = 2, partition = `noniid-labeldir`, beta = 0.5
* n_parties = 10, partition = `noniid-labeldir`, beta = 0.5
* n_parties = 10, partition = `noniid-labeldir`, beta = 0.1
* n_parties = 10, partition = `noniid-#label1`
* n_parties = 10, partition = `iid-diff-quantity`, beta = 0.5

You can run the scripts `partitions/partition[1-6].sh` to generate the partitions.

### Step 3
You can run FedTree with the partitioned dataset and the configuration file. An example of the configuration file is shown below.

```angular2html
data=./data/creditcard1/0.csv,./data/creditcard1/1.csv
test_data=./data/creditcard_test.csv
n_parties=2
num_class=2
mode=horizontal
objective=binary:logistic
data_format=csv
privacy_tech=none
model_path=fedtree.model
max_num_bin=16
learning_rate=0.1
max_depth=6
n_trees=10
```

Then, you can run the following command to train the model.

```angular2html
./build/bin/FedTree-train example.conf
```
The expected output is ``AUC = 0.85367``, which is the AUROC of the test dataset.

## Metric
The main focus of the challenge is to achieve high AUC for each setting. We will take the average of the AUC of the six settings as the final score.
Besides AUC, we also take efficiency (e.g., the running time of your training) into consideration. You are encouraged to develop any techniques to improve the accuracy and efficiency.

## Requirement
You can modify any tree-related parameters (e.g., `learning_rate`, `n_trees`) to achieve the best AUC. 
You can also do any feature engineering work for each partitioned dataset, as long as the data information of each party is kept locally.
You may refer to [FedTree documentation](https://fedtree.readthedocs.io/en/latest/Parameters.html#parameters-for-gbdts) for a full list of parameters.
Moreover, you can modify the source code of FedTree if you want to implement any new tree-based algorithms or functionalities.
The goal is to achieve the best AUC for each federated setting listed in Step 2 of Setup.


## Submission
You need to submit a zip file including 1) a report to briefly describe your understanding of the problem, your solution, results, and findings (<= 2 pages); 2) the configuration file for each setting; 3) any source code you have implemented for your solution.
The code of FedTree does not need to submit if you do not modify it.
Please email your submission to `liqinbin1998@gmail.com`.