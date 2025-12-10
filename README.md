# MagiaPose

### 安装依赖：

使用 requirements.txt
```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple some-package
```

或者使用 environment.yml
```bash
conda env create -f environment.yml
```

Install Mamba:

```
MAMBA_FORCE_BUILD=TRUE pip install mamba-ssm==2.2.6.post3 --no-cache-dir -vvvvvvvvvv --no-build-isolation
```

### 训练

0. 准备预训练权重

在 models/yolov8s 文件夹下准备 yolov8s 预训练权重：`wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt`。
在 models 文件夹下准备 vivit 预训练权重：`git clone https://hf-mirror.com/google/vivit-b-16x2-kinetics400 `。

1. 准备数据集

2. 生成 train.txt 和 val.txt

运行 `python scripts/gen_filelist.py`。

3. 创建数据缓存

运行 `PYTHONPATH=. CUDA_VISIBLE_DEVICE=0 python src/data/video_dataset.py`。

4. 训练

单机单卡：`PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python src/train.py trainer=gpu`。
单机多卡：`PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 python src/train.py`。
