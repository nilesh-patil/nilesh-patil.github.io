---
layout: single
title: "Galactic Morphology using Deep-Learning"
date: 2017-07-25T15:39:55-04:00
last_modified_at: 2017-07-25T19:48:19-04:00
categories: [blog]
tags: [deep learning, galaxy, computer-vision, machine learning, neural networks]
excerpt: "Training a deep neural net to understand galactic structure"
math: true
header:
  overlay_image: /images/blog/feature/galaxy.jpg
  overlay_filter: 0.5
  teaser: /images/blog/feature/galaxy.jpg
  caption: "Galaxy morphology classification"
redirect_from:
  - /blog/galactic-morphology-using-deep-learning/
---
## Introduction

Astronomy has historically been one of the most data intensive fields & a major chunk of this data is collected as images collected by a number of telescopes - terrestrial as well as in space. A BIG data-project which aims to collate this data from various sources to form a coherent picture of the universe is [Sloan Digital Sky Survey](http://www.sdss.org/).


To quote the project website:

> The Sloan Digital Sky Survey has created the most detailed three-dimensional maps of the Universe ever made, with deep multi-color images of one third of the sky, and spectra for more than three million astronomical objects. Learn and explore all phases and surveys — past, present, and future — of the SDSS.


A citizen science project called [Galaxy zoo](https://www.galaxyzoo.org) was launched in 2007, through this project thousands of volunteers classified 100k+ images of galaxies. A flow-chart of questions asked to volunteers shown on the project website is as follows.

![Galaxy Zoo decision tree](/images/blog/galaxyzoo/00.galaxyzoo-tree.png){: .center-image height="850px" width="1050px"}


## Data Description

The dataset consists of 100k+ jpeg images and the corresponding score vector for each image. The score vector has 37 values where each value represents the weighted score from volunteers in the project.

The important point to remember is that the scores aren't probability score per se. They are weighted scores & so they vary from 0 to 1 but all sub scores for a question don't necessarily sum up to 1 as a rule.

Each image is of size `424×424×3` and each value is between `0`–`255`. A good practice is to rescale the data. In our experiments, we normalize the images by computing $\mu_\text{channel}$ and $\sigma_\text{channel}$ over the full dataset, then normalizing each channel using its corresponding $\mu$ and $\sigma$. The channels and cell values do not represent the physical aspect of data collection — they are standardized to the accepted image format range of `0`–`255` — so the normalization is more of a hack for better gradient updates than a domain-knowledge-based modification.

A few sample images from the dataset:

![Galaxy sample](/images/blog/galaxyzoo/01.galaxies.png){: .center-image}

These images are read in as `numpy` arrays in python with the following representation:

![Numpy array](/images/blog/galaxyzoo/02.numpy_array.png){: .center-image height="250px" width="350px"}

## Fully Convolutional Classifier

A convolutional network takes in your image array as input extracts features from this array which best represents the task at hand & then gives out a classification/regression output. Standard classification models use one-vs-rest scheme to represent output for an elegant representation of the classification task. In this form, the correct class is assigned `1` while other possible classes in the dataset are assigned `0`. The output vector is of length `c`, where `c` is total number of classes in the dataset.

In the Galaxy Morphology classification task, we use standard `.jpeg` images to learn the shape attributes as a vector of length 37 which describes its properties. We set it up as a regression task in this case, since our ground truth is a weighted version of votes gathered from volunteers.

The model takes in a normalized array representing input image. This array passes through the following layers stacked after each other:

1. *Convolutional layer* : It consists of a set of learnt features. In terms of standard modeling terminology, the features that a model uses are usually handcrafted i.e. some form of transformations of the raw input data. In images, the kernels that form the convolutional layer are expected to learn optimal features for the task at hand, instead of features crafted by a domain expert. Since the output from this convolutional layer is learnt w.r.t output, the features being generated at each step should ideally be the optimal representation of input provided at that step.

2. *Pooling layer* : The pooling layer reduce size of incoming representation by selecting from a set of appropriate downsampling functions. The `max-pooling` layer chooses maximum from a given volume of array as an appropriate representation of the focus. Similarly, `average-pooling` takes average of the volume.

3. *Activation* : Activation functions are used to introduce non-linearity in the model. This layer applies a given function to each element of input array. Standard activation functions used in models are 'relu', 'softmax', 'sigmoid', 'tanh' etc.

4. *Dropout*: a regularization technique developed to reduce overfitting in deep neural networks. At training time, the activations of a randomly chosen fraction of neurons are set to `zero`; at prediction time, the weights learnt for those units are multiplied by the keep-probability `p`. Below, (a) shows a standard fully-connected network and (b) shows the same network with dropout applied — at each training step a different subset of activations is zeroed, which prevents any single neuron from dominating the learned representation.

   ![Dropout network](/images/blog/galaxyzoo/03.droput_representation.png){: .center-image height="250px" width="500px"}

5. *Batch normalization*: a major problem during training is that the distribution of inputs to each successive layer shifts as the parameters of preceding layers update. This *internal covariate shift* forces small learning rates and slow convergence. Batch normalization addresses it by normalizing each channel's activations to $\mu_x = 0$ and $\sigma_x = 1$ across the current mini-batch, then applying a learned affine transform:

   $$X_\text{out} = \gamma \cdot \frac{X_\text{in} - \mu_X}{\sigma_X} + \beta$$

   Here $\mu_X$ and $\sigma_X$ are computed channelwise. The effect is that subsequent layers see a more stable input distribution, which permits larger learning rates and faster training.

### Setup

The layers above are stacked into a module and we experiment with several network structures, starting from well-established architectures and moving toward more recently published ones. The final layers are `Dense` (fully connected) layers; their output is compared with the expected output to compute the loss for each observation. The loss is back-propagated to update the convolutional kernel weights and the dense layer weights, with the expectation that, given enough data, the network learns features that are optimal for the task at hand.

In our setup, features are extracted successively from the raw galaxy image and a regression head matches the human-generated score vector for the same image. The score is a vector of length `37`, where each entry encodes one physical aspect of the galaxy — together describing its morphology.

## References

1. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf). *Journal of Machine Learning Research*, 15.
2. Ioffe, S. & Szegedy, C. (2015). [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167). *Proceedings of the 32nd International Conference on Machine Learning*.
