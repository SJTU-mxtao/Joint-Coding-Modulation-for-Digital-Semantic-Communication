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

## Results
1. Accuracy
   
| SNR (dB) | BPSK | 4QAM | 16QAM | 64QAM |
| :-----: | :----: | :----: | :----: | :----: |
| 18 | 0.8783 | 0.8705 | 0.8723 | 0.8750 |
| 12 | 0.8713 | 0.869 | 0.8765 | 0.8730 |
| 6 | 0.8771 | 0.8682 | 0.8768 | 0.8688 |
| 0 | 0.8777 | 0.8726 | 0.8732 | 0.8772 |
| -6 | 0.8683 | 0.8711 | 0.8737 | 0.8738 |
| -12 | 0.7735 | 0.8674 | 0.8810 | 0.8645 |
| -18 | 0.4316 | 0.6148 | 0.6320 | 0.6271 |

2. PSNR (dB)
   
| SNR (dB) | BPSK | 4QAM | 16QAM | 64QAM |
| :-----: | :----: | :----: | :----: | :----: |
| 18 | 19.5620 | 21.4954 | 23.6771 | 25.0308 |
| 12 | 19.4385 | 21.2107 | 23.4145 | 24.2522 |
| 6 | 19.1311 | 20.7429 | 22.2025 | 22.5085 |
| 0 | 17.5842 | 18.7376 | 19.7264 | 19.7710 |
| -6 | 15.2930 | 16.4231 | 16.7817 | 16.9260 |
| -12 | 13.1355 | 13.6273 | 13.6800 | 13.7743 |
| -18 | 12.9488 | 13.0387 | 13.0450 | 13.0472 |
