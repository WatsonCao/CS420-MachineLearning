# CS420-MachineLearning
Our DTN project is based on [Unsupervised Cross-Domain Image Generation.](https://arxiv.org/abs/1611.02200)
Because the update of certain libraries, some error happened during the re-implementation of paper code. Thanks for the reference of [yunjey](https://github.com/yunjey) and [SrCMpink](https://github.com/SrCMpink)

#### First of all, you need to download SVNH
```bash
$ mkdir -p svhn
$ wget -O svhn/train_32x32.mat http://ufldl.stanford.edu/housenumbers/train_32x32.mat
$ wget -O svhn/test_32x32.mat http://ufldl.stanford.edu/housenumbers/test_32x32.mat
$ wget -O svhn/extra_32x32.mat http://ufldl.stanford.edu/housenumbers/extra_32x32.mat
```
#### Resize MNIST dataset to 32x32 
```bash
$ python prepro.py
```

#### Pretrain the model f
```bash
$ python main.py --mode='pretrain'
```

#### Train the model G and D
```bash
$ python main.py --mode='train'
```

#### Transfer SVHN to MNIST
```bash
$ python main.py --mode='eval'
```
