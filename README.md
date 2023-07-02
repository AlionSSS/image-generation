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
