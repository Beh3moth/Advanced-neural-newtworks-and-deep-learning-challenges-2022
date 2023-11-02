# Advanced-neural-newtworks-and-deep-learning-challenges-2022

**Group nickname:** NotRunningCode

**Students:** 
- Federica Baracchi
- Federico Camilletti
- Patricks Francisco Tapia Maldonado


## Homework 1 (28 November 2022)

### Image Classification

**Leaderboard nickname:** NotRunningCode

**Students:** 
- Federica Baracchi
- Federico Camilletti
- Patricks Francisco Tapia Maldonado

### 1 Data

#### 1.1 Data Augmentation

In our analysis of the dataset, we identified an imbalance within the dataset, which comprised 3542 images divided into 8 classes, each with varying numbers of images. To address this issue, we opted for over-sampling using data augmentation via the Keras library. We experimented with both hard and light augmentation, with the latter showing better accuracy on the test set. Despite this, the underrepresented species continued to pose challenges, leading us to adapt the receptive field of the neural network for optimal performance.

#### 1.2 Splitting the Data Set

We split the dataset into an 80% training set and a 20% test set, further allocating 20% of the training set for validation. This division proved effective not only during training but also in producing results comparable to the hidden set.

### 2 Convolutional Neural Networks

Our initial approach to plant classification involved sequential blocks of layers comprising convolutions, poolings, and dropouts, the latter used to prevent overfitting. We found that using average pooling instead of max pooling contributed to the model's accuracy, leading us to use average pooling exclusively. Further enhancements were made, culminating in a model with 4 blocks of convolutions and average pooling (32 → 64 → 128 → 256) and a single classifier with 512 neurons in the fully connected layers. Our experiments with double convolution in each block and the use of non-relu activation functions resulted in decreased accuracy.

### 3 Transfer Learning

In our pursuit of improving model accuracy, we implemented various renowned CNN architectures such as VGG16/VGG19, ResNet50, EfficientNet, Xception, and ConvNeXt, ultimately finding success with the ConvNeXtXLarge model. We adapted this model, incorporating the GlobalAveragePooling layer, freezing CNN weights while training only FCN layers, and employing fine-tuning techniques. To counter overfitting, we introduced the dropout technique, randomly deactivating thirty percent of neurons for each epoch. Our efforts to address class imbalances through proportional weighting were ultimately not effective in this model.


## Homework 2 (23 December 2022)

### 1 Introduction

The aim of this homework was to correctly classify samples in the multivariate time series format. Several models were exploited during the challenge to achieve the best result in terms of accuracy. This report first describes the data augmentation problem and then delves into the five models used: ResNet, Bilateral-LSTM, Ensemble learning, and K-fold cross-validation. The last section is dedicated to the 1D-CNN model, which ultimately yielded the best result.

### 2 Data

The dataset comprises 2429 samples, each composed of 36 timestamps and 6 features, with 12 classes. As depicted in the image below, the dataset is unbalanced with respect to the classes, necessitating data augmentation. Various approaches were experimented with, including scaling factors, random noise addition, and shifts of timestamps to feed the training with more time series-consistent data. The best data augmentation method involved a combination of shifts and random noise. An external library was imported to randomly shift the timestamps, effectively doubling the training set. A further shift with added random noise was applied to the entire training set, resulting in the training of the model with 7772 samples.

To better understand the dataset, an analysis of the correlation between the 6 features was conducted. It was observed that the features are closely related to each other, except for feature 1. An attempt was made to reduce the number of features using Principal Component Analysis (PCA). However, it did not yield significant improvements, and all 6 features were ultimately used.

## 3 Models

#### 3.1 ResNet

Building upon the success of the 1D convolutional model, we implemented ResNet using the reshape filter of Keras to create an image from each pixel. The resulting model consists of a total of 510,988 parameters (508,428 trainable and 2,560 non-trainable). Although the accuracy on the validation set during training reached 75%, it scored around 69% in the hidden test of phase one. Efforts were made to improve the results, particularly by addressing overfitting, but with limited success.

#### 3.2 Bilateral-LSTM

Both a normal LSTM and a bilateral LSTM were tested during the challenge, but their results were below the average of our models.

#### 3.3 Ensemble Learning

Due to similar confusion matrices and the failure to recognize some classes, we attempted ensemble training. This involved training a single 1D convolutional model for each class and then combining the submodels into a single model. However, this method did not resolve the issue of unrecognized classes, as the submodels performed similarly.

#### 3.4 K-Fold Cross Validation

K-Fold Cross Validation was tested during the work as a resampling procedure commonly used to evaluate machine learning models. However, its efficiency varied among different models, leading to the decision not to use it for the final model.

### 4 Final Model: 1D-CNN

The most effective model in terms of error reduction and accuracy was the 1D Convolutional Neural Network. The model employed convolution and average pooling techniques iterated three times. For the convolution, a kernel size of 2 and filter sizes of 128, 256, 512 were used. "Relu" was employed as the activation function, and global average pooling was used to obtain the feature vector of the data instead of "flatten."

To mitigate overfitting, the following techniques were implemented:

1. Dropout was applied three times, deactivating 20% of neurons.
2. A L2 regularization term (ridge regularization) with λ = 0.01 was added to the FC layer.

The loss curve evaluated on the train test closely matched that evaluated on the validation set. The confusion matrix revealed the best-identified classes, with classes 11, 7, and 6 showing the highest accuracy. However, it was observed that the model occasionally confused class 11 with class 8 and vice versa.
