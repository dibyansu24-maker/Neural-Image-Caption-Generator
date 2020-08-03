# Neural-Image-Caption-Generator
Here I have implemented a first-cut solution to the Image Captioning Problem, i.e. Generating Captions for the given Images using Deep Learning methods.

## Abstract

Automatically describing the content of an image fundamental problem in artificial intelligence that connects computer vision and natural language processing. Being able to build a model that could bridge these two fields will help us apply various techniques of each other to solve a fundamental problem in artificial intelligence. The paper which we worked on is [Show and tell: A neural image caption generator [et vinyals]](http://research.google.com/pubs/archive/43274.pdf) where we will use CNN and Recurrent neural networks such as LSTM to build an end to end model.

Keywords: NLP,Vision,CNN,RNN,LSTM

## Table of Contents
* Abstract
* Introduction
* Model Architecture
* Dataset(Flickr 8k)
* Training
* Inference
* Results
* Scope of Further Improvements
* References

## Introduction

In the area of artificial intelligence, automatically describing the content of an image using properly formed English sentences is a challenging task. Leveraging the advances in recognition of objects allows us to drive natural language generation systems, but the current approaches have limited ability in their expressivity. For this paper ([Show and tell- A Neural Image Caption Generator](http://research.google.com/pubs/archive/43274.pdf)), the authors combined deep CNN for image classification with RNN for sequence modelling to create a single end to end neural network that generates a description of images. They take the image I as input and produced a target sequence of words S = {S1, S2, ....} by directly maximizing the likelihood p(S|I). They used a deep CNN as an encoder function to produce a rich representation of the input image by embedding it to a fixed-length vector. This embedding vector will be used as the initial hidden state of a decoder RNN that will be used to generate the target sentence. They present an end-to-end system for this sentence caption generation problem. Their neural network is trainable by using basic Stochastic gradient descent algorithm or any other flavour of gradient descent. Finally, through experimental results, they show their method could perform significantly better than current (at that time now we also have attention models) state-of-art approaches.

## Model Architecture
The goal is to directly maximize the probability of the correct description given the image by:
<img width="362" alt="Annotation 2020-08-03 152242" src="https://user-images.githubusercontent.com/64517601/89170414-38568f00-d59d-11ea-9de7-97badf02f6be.png">

**Equation explanation:**
1. (I<sup>i</sup>,S<sup>i</sup>) is image-caption pair
1. D is training dataset 
1. θ is the parameter of our model
Now we will use chain rule over the above equation to model the joint probability logp(S|I,θ)over S, we could get :
<img width="400" alt="Annotation 2020-08-03 153156" src="https://user-images.githubusercontent.com/64517601/89171236-7e602280-d59e-11ea-803d-51defe4ffb65.png">
Sₜ is the t th word in the caption S. So the authors model the conditional probability using the LSTM(512 units), which is a special form of recurrent neural network.To be more specific, at the time step t-1, treat the hidden h(t-1) as a non-linear representation of I,θ, S0,........, Sₜ-2 and given the word Sₜ-1, then calculate h(t-1)= f(h(t-2), S(t-1)).Finally, model p(Sₜ|I,θ,Sо,..........,Sₜ-1) using pt = Softmax(h(t-1)). The p is the conditional probability distribution over the whole dictionary, which suggests the word generated at time step t. One more thing that needs to be specifically addressed here is that authors used CNN to initialize S0. In our attempt to replicate the paper we have used InceptionV3 to get pre-trained weights and encode image vectors to initialize LSTM.

## Dataset(Flickr 8k)
Due to limited computation power, we will train our model on Flickr 8k dataset.<br>
This dataset contains 8000 images. We have divided our dataset in the following fashion:<br>
Training set:6000<br>
Validation/Dev set=1000<br>
Test set=1000<br>
**Link for downloading Dataset:** [Flickr 8k Dataset Folder](https://drive.google.com/drive/folders/1VOUAUphQN7jKPkS0fVPcRHhwHWBzbvk3?usp=sharing)<br>
**Exploring dataset:** Flickr 8k dataset contains 8092 images and up to five captions for each image.<br>
For simplicity, we have provided the google drive link for the dataset. Moreover in the drive folder only we have separated and made Training, Validation and Test set.<br>
**Random dataset**: We created our random dataset too, which has 30-40 random images from the internet to check our model efficiency in the real world. We will attach an analysis over that set too in our report.<br>
Below is an example from Flickr 8k dataset:<br>
<img width="210" alt="Example Image" src="https://user-images.githubusercontent.com/64517601/89172296-05fa6100-d5a0-11ea-835d-7d11c18e93ca.jpg"> <img width="665" alt="Annotation 2020-07-16 220339" src="https://user-images.githubusercontent.com/64517601/89173089-532b0280-d5a1-11ea-976f-88c5bef24053.png">
