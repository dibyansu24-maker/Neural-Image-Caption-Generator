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
θ?= arg maxθ∑(I,S)logp(S|I;θ)
