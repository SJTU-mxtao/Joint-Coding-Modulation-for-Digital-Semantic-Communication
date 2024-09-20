# Python code for "Joint coding-modulation for digital semantic communications via variational autoencoder"
This repository contains the original code and models for the work _Joint Coding-Modulation for Digital Semantic Communications via Variational Autoencoder_[1].

[1] Y. Bo, Y. Duan, S. Shao and M. Tao, "Joint Coding-Modulation for Digital Semantic Communications via Variational Autoencoder," in IEEE Transactions on Communications, doi: 10.1109/TCOMM.2024.3386577.

## Requirements
* matplotlib==3.7.2
* numpy==1.23.5
* pandas==2.0.3
* scikit_learn==1.3.0
* scipy==1.13.0
* scikit-image==0.21.0
* torch==1.12.1+cu113
* torchvision==0.13.1+cu113
* tqdm==4.65.0

## Training & Evaluation
This code implements 4 modulation schemes: BPSK, 4QAM, 16QAM and 64QAM. 

For training, run the following command (as an example):
```
python main.py --mode 'train' --mod_method '64qam' --load_checkpoint 1
```

For evaluation, run the following command (as an example):
```
python main.py --mode 'test' --mod_method '64qam' --load_checkpoint 1
```
