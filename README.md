# CNN Architectures for Image Classification

[![Packagist](https://img.shields.io/packagist/l/doctrine/orm.svg)](LICENSE.md)
---


### Author
Arpit Aggarwal


### Introduction to the Project
In this project, different CNN Architectures like AlexNet, VGG-16, VGG-19, InceptionNet, DenseNet-121 and ResNet-50 were used for the task of Dog-Cat image classification. The input to the CNN networks was a (224 x 224 x 3) image and the number of classes were 2, where '0' was for a cat and '1' was for a dog. The CNN architectures were implemented in PyTorch and the loss function was Cross Entropy Loss. The hyperparameters to be tuned were: Number of epochs(e), Learning Rate(lr), momentum(m), weight decay(wd) and batch size(bs). 


### Data
The data for the task of Dog-Cat image classification can be downloaded from: https://drive.google.com/drive/folders/1EdVqRCT1NSYT6Ge-SvAIu7R5i9Og2tiO?usp=sharing. The dataset has been divided into three sets: Training data, Validation data and Testing data. The analysis of different CNN architectures for Dog-Cat image classification was done on comparing the Training Accuracy and Validation Accuracy values.


### Results
The results after using different CNN architectures are given below:

1. <b>AlexNet</b><br>
Training Accuracy = 96.64% and Validation Accuracy = 93.59% (e = 50, lr = 0.005, m = 0.9, bs = 64, wd = 0.001)

2. <b>VGG-16</b><br>
Training Accuracy = 97.55% and Validation Accuracy = 95.73% (e = 50, lr = 0.005, m = 0.9, bs = 32, wd = 0.001)

3. <b>VGG-19</b><br>
Training Accuracy = 91.5% and Validation Accuracy = 92.7% (e = 45, lr = 0.005, m = 0.9, bs = 32, wd = 5e-4)

4. <b>DenseNet-121</b><br>
Training Accuracy = 99.2% and Validation Accuracy = 92.9% (e = 60, lr = 0.003, m = 0.9, bs = 32, wd = 0.001)

5. <b>InceptionNet</b><br>
Training Accuracy = 98.1% and Validation Accuracy = 93.8% (e = 60, lr = 0.003, m = 0.9, bs = 32, wd = 0.001)

6. <b>ResNet-50</b><br>
Training Accuracy = 92.5% and Validation Accuracy = 92.3% (e = 60, lr = 0.01, m = 0.9, bs = 32, wd = 0.001)


### Software Required
To run the jupyter notebooks, use Python 3. Standard libraries like Numpy and PyTorch are used.


### Credits
The following links were helpful for this project:
1. https://www.youtube.com/channel/UC88RC_4egFjV9jfjBHwDuvg
2. https://github.com/pytorch/tutorials
