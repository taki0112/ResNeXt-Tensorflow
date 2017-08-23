# ResNeXt-Tensorflow
Tensorflow implementation of [ResNeXt](https://arxiv.org/abs/1611.05431) using **Cifar10**

If you want to see the ***original author's code***, please refer to this [link](https://github.com/facebookresearch/ResNeXt)

## Requirements
* Tensorflow 1.x
* Python 3.x
* tflearn (If you are easy to use ***global average pooling***, you should install ***tflearn***

## Compare Architecture
### ResNet
![ResNet](./assests/ResNet.JPG)

### ResNeXt
![ResNeXt](./assests/ResNeXt.JPG)

* I implemented (b) 
* (b) is ***split + transform(bottleneck) + concatenate + transition + merge***

## Comapre Results (ResNet, DenseNet, ResNeXt)
![compare](./assests/comparision.png)

## Related works
* [DenseNet](https://github.com/taki0112/Densenet-Tensorflow)

## Author
Junho Kim
