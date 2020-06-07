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

1. AlexNet
Training Accuracy = 90.1% and Validation Accuracy = 87.9% (e = 50, lr = 0.005, m = 0.9, bs = 64, wd = 5e-4)

2. VGG-16
Training Accuracy = 93.1% and Validation Accuracy = 92.6% (e = 50, lr = 0.005, m = 0.9, bs = 32, wd = 5e-4)

3. VGG-19
Training Accuracy = 91.5% and Validation Accuracy = 92.7% (e = 45, lr = 0.005, m = 0.9, bs = 32, wd = 5e-4)

4. DenseNet-121
Training Accuracy = 99.2% and Validation Accuracy = 92.9% (e = 60, lr = 0.003, m = 0.9, bs = 32, wd = 0.001)

5. InceptionNet
Training Accuracy = 98.1% and Validation Accuracy = 93.8% (e = 60, lr = 0.003, m = 0.9, bs = 32, wd = 0.001)

6. ResNet-50
Training Accuracy = 92.5% and Validation Accuracy = 92.3% (e = 60, lr = 0.01, m = 0.9, bs = 32, wd = 0.001)


### Software Required
To run the jupyter notebooks, use Python 3. Standard libraries like Numpy and PyTorch are used.
