#!/bin/bash

#train the baseline on various datasets (excluding CIFAR, 1. these were already done)

#train pretrained model (CIFAR) on the dataset OIDv6
#gives history_CIFAR-OIDv6.npy
#      baseline_CIFAR-OIDv6.h5
#      results_CIFAR-OIDv6.h5
python3 train_CIFAR-OIDv6.py baseline_CIFAR.h5 64 100


#train CIFAR-OIDv6 on OA-datasets
#each gives history_OA.npy
#           baseline_OA.h5
#           results_OA.txt
python3 train_OA.py baseline_CIFAR-OIDv6.h5 64 100 0
mv history_OA.npy history_OAR2.npy
mv baseline_OA.h5 baseline_OAR2.h5
mv results_OA.txt results_OAR2.txt

python3 train_OA.py baseline_CIFAR-OIDv6.h5 64 100 1
mv history_OA.npy history_OAH2.npy
mv baseline_OA.h5 baseline_OAH2.h5
mv results_OA.txt results_OAH2.txt


#train OIDv6 on OA-datasets
#each gives history_OA.npy
#           baseline_OA.h5
#           results_OA.txt
python3 train_OA.py baseline_OIDv6.h5 64 100 0
mv history_OA.npy history_OAR3.npy
mv baseline_OA.h5 baseline_OAR3.h5
mv results_OA.txt results_OAR3.txt

python3 train_OA.py baseline_OIDv6.h5 64 100 1
mv history_OA.npy history_OAH3.npy
mv baseline_OA.h5 baseline_OAH3.h5
mv results_OA.txt results_OAH3.txt
