# SinglePrompt (CVPRF 2026)

Pytorch Implementation for [Is Prompt Selection Necessary for Task-Free Online Continual Learning?]

## Install
```bash
conda env create -f environment.yml
conda activate tfocl
```

##  How to Run

###  1. Training without Buffer

#### CIFAR100
```bash
bash scripts/buffer0_cifar.sh
```
#### Tiny Imagenet
```bash
bash scripts/buffer0_tiny.sh
```
#### Imagent-R
```bash
bash scripts/buffer0_imgr.sh
```

---

###  2. Buffer-based Training

Supervised training enhanced with memory replay (buffer).

#### CIFAR100
```bash
bash scripts/buffer{MEM_SIZE}_cifar.sh
```
#### Tiny Imagenet
```bash
bash scripts/buffer{MEM_SIZE}_tiny.sh
```
#### Imagent-R
```bash
bash scripts/buffer{MEM_SIZE}_imgr.sh
```

Modify the following arguments in the script:

- **Buffer size**  
  Change `MEM_SIZE` to control the buffer size.  
  Examples:
  - `MEM_SIZE=500`
  - `MEM_SIZE=2000`
 
 ---

 ### Acknowledgement
This implementation is developed based on the source code of [MISA](https://github.com/kangzhiq/MISA), [MVP](https://github.com/KU-VGI/Si-Blurry).
