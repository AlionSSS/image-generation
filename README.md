# image-generation
- This is an AI project that uses GAN to generate images.
- models 
  - [x] DCGAN
  - [ ] ResDCGAN(DCGAN + ResidualBlock)
  - [ ] xxx

## Environment
- 主要使用 `PyTorch 1.12.1`、`Python 3.7.12`
- 依赖库详见 [requirements.txt](requirements.txt)

## Dataset 
- **Anime Faces Dataset 128**
  1. 请先下载数据集 [Anime Faces Dataset 128](https://www.kaggle.com/datasets/dimensi0n/anime-faces-dataset) 
  2. 将下载好的文件`archive.zip`解压至项目的`./dataset`下
  3. 存储路径格式如`./dataset/AnimeFaces128/face/*.png`

## Help Info
- 命令示例
```shell
python main.py help
```

## Model Train
- 先启动 Visdom Server，见 [Run Visdom](#run-visdom)
- 命令示例
```shell
# Linux
python main.py train \
# --netg-path='checkpoints/GeneratorNet_100.pth' \
# --netd-path='checkpoints/DiscriminatorNet_100.pth' \
--data-path='dataset/AnimeFaces128' \ 
--max-epoch=200 \
--batch-size=256 \
--lr-g=0.002 \
--lr-d=0.0002 \
# 是否使用Visdom，默认为True
--vis=True \
# visdom 环境名称
--visdom-env='image-generation'
```

## Generate Image
- 命令示例
```shell
# Linux
python main.py generate \
--netg-path='checkpoints/GeneratorNet_200.pth' \
--netd-path='checkpoints/DiscriminatorNet_200.pth' \
--gen-img='result.png' \
--gen-num=64
```
- 生成图像的结果见`--gen-img`参数指定的文件，如`result.png`

## Run Visdom
- 命令示例
```shell
# 阻塞启动
python -m visdom.server

# 非阻塞启动
nohup python -m visdom.server &
```
- 启动后即可使用Web浏览器访问 [http://localhost:8097](http://localhost:8097)
- 在网页选择环境`image-generation`，通过图表查看训练过程中的`picture`、`error`
![screenshot-2023-07-02 152247.png](resource%2Fscreenshot-2023-07-02%20152247.png)