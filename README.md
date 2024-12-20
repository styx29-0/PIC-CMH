# PIC-CMH
The source code for the paper "Prompt-Infused Continual Cross-Modal Hashing".

## datasets & pre-trained cmh models
1. Download datasets MSCOCO and NUSWIDE

```
MSCOCO
url: https://pan.baidu.com/s/17KpPah2PKLdlukwPA7Za3A?pwd=2024
code: 2024

NUSWIDE
url: https://pan.baidu.com/s/1-8DHjq9Ap2NplZy07_5CEA?pwd=2024
code: 2024
```

2. Change the value of `data_path` in file `./config.yaml` to `/path/to/dataset`.

## python environment
``` bash
conda create -n PIC_CMH python=3.8
conda activate PIC_CMH
pip install -r requirements.txt
```

## training
``` python
python main.py
```