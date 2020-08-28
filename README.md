# Neural Image Caption Generator

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
### **NOTE:** We could not provide the Glove file in the repo, you can download it from [here](https://www.kaggle.com/mrugankakarte/glove-6b-200d-pretrained-word-vectors/download)

## Introduction

In the area of artificial intelligence, automatically describing the content of an image using properly formed English sentences is a challenging task. Leveraging the advances in recognition of objects allows us to drive natural language generation systems, but the current approaches have limited ability in their expressivity. For this paper ([Show and tell- A Neural Image Caption Generator](http://research.google.com/pubs/archive/43274.pdf)), the authors combined deep CNN for image classification with RNN for sequence modelling to create a single end to end neural network that generates a description of images. They take the image I as input and produced a target sequence of words S = {S1, S2, ....} by directly maximizing the likelihood p(S|I). They used a deep CNN as an encoder function to produce a rich representation of the input image by embedding it to a fixed-length vector. This embedding vector will be used as the initial hidden state of a decoder RNN that will be used to generate the target sentence. They present an end-to-end system for this sentence caption generation problem. Their neural network is trainable by using basic Stochastic gradient descent algorithm or any other flavour of gradient descent. Finally, through experimental results, they show their method could perform significantly better than current (at that time now we also have attention models) state-of-art approaches.

## Model Architecture
The goal is to directly maximize the probability of the correct description given the image by:
<img width="362" alt="Annotation 2020-08-03 152242" src="https://user-images.githubusercontent.com/64517601/89170414-38568f00-d59d-11ea-9de7-97badf02f6be.png">

**Equation explanation:**
1. (I<sup>i</sup>,S<sup>i</sup>) is image-caption pair
1. D is training dataset 
1. θ is the parameter of our model
Now we will use chain rule over the above equation to model the joint probability logp(S|I,θ)over S, we could get: <br><img width="400" alt="Annotation 2020-08-03 153156" src="https://user-images.githubusercontent.com/64517601/89171236-7e602280-d59e-11ea-803d-51defe4ffb65.png"><br> Sₜ is the t th word in the caption S. So the authors model the conditional probability using the LSTM(512 units), which is a special form of recurrent neural network.To be more specific, at the time step t-1, treat the hidden h(t-1) as a non-linear representation of I,θ, S0,........, Sₜ-2 and given the word Sₜ-1, then calculate h(t-1)= f(h(t-2), S(t-1)).Finally, model p(Sₜ|I,θ,Sо,..........,Sₜ-1) using pt = Softmax(h(t-1)). The p is the conditional probability distribution over the whole dictionary, which suggests the word generated at time step t. One more thing that needs to be specifically addressed here is that authors used CNN to initialize S0. In our attempt to replicate the paper we have used InceptionV3 to get pre-trained weights and encode image vectors to initialize LSTM.

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
<img width="210" alt="Example Image" src="https://user-images.githubusercontent.com/64517601/89172296-05fa6100-d5a0-11ea-835d-7d11c18e93ca.jpg"> <img width="665" alt="Annotation 2020-07-16 220339" src="https://user-images.githubusercontent.com/64517601/89173089-532b0280-d5a1-11ea-976f-88c5bef24053.png"><br>

## Training
### 1. Dataset Preprocessing:
  * Converting all letters to lowercase and removing hanging ‘s’ and ‘a’.
  * We then made a vocabulary of unique words which occurred for at least 10 times(threshold frequency). We got a vocabulary size of 1651 after this process. Then, <<START>>,<<END>> and padding tokens were added. 
  * A dictionary was also made for all descriptions.
  * Images were also preprocessed as per the requirements for the input of InceptionV3 model i.e. reshaped into (sample_size, 299, 299, 3) and were normalized to [-1,1]
### 2. Actual Training:
  * Actual training can be divided into two parts: encoding images through CNN and caption generation through LSTM.
  * **Image encoding (Extracting features from images)**: We used Inception V3 model pre-trained on ImageNet as image encoder, last layer weights are used to encode our images. This encoding will be our hidden state input for the LSTM layer at the time step 0.
  * **Caption Generation**: We take each word and put it through the word embedding layer (Embedding dimensions: 200d) which produces a word processing vector. Then we send this word embedding to the LSTM cell to predict the next word. The procedure can be mathematically described as follows:<br> <img width="350" alt="Annotation 2020-08-03 160850" src="https://user-images.githubusercontent.com/64517601/89174548-a7cf7d00-d5a3-11ea-8605-d04d83b124f2.png">
  * **Loss and Metric**: The loss is calculated as follows: <br> <img width="400" alt="Annotation 2020-08-03 164629" src="https://user-images.githubusercontent.com/64517601/89177364-e9165b80-d5a8-11ea-9d92-dc91ad278b8e.png"> <br>We use the training loss as a criterion to terminate the training process, that is when there is no significant improvement in the training loss we stop the training. We mainly use BLEU-4 scores, which are calculated on the test datasets, as the criteria to evaluate the quality of the generated captions, where the captions are generated based on the maximal probability rule.<br>
  **Model Architecture:**<br> <img width="450" height="400" alt="Annotation 2020-08-03 162036" src="https://user-images.githubusercontent.com/64517601/89175387-4f00e400-d5a5-11ea-8d96-4f32ad1f22f4.png"><img width="272" height="400" alt="Annotation 2020-08-03 162525" src="https://user-images.githubusercontent.com/64517601/89177003-31814980-d5a8-11ea-8b61-74493db193f9.png">
  
## Experimental Analysis(Hyperparameter Tuning)
We tried many hyperparameter combinations. We tried to reduce variance (the difference between training loss and validation loss) and bias (high error). We tried optimum hyperparameters to achieve low variance and low bias.

After trying a various number of epochs size and batch size we found that smaller batch size and epoch numbers were able to reduce overfitting. Since the dataset is limited and objective is to caption real-world images overgeneralization and overfitting were a major issue. This paper [On Large-batch Training For Deep Learning: Generalization Gap And Sharp Minima](https://arxiv.org/pdf/1609.04836) can be referred to understand why smaller batch sizes help in avoiding sharp minima and overgeneralization.
We concluded our batch_size to be 30/32 (batch size of power of 2 are preferred)
No. of epochs was found to be optimum around 8-12.
Adam optimizer is used. Two learning rates are discussed 0.001 and 0.0005
Categorical cross-entropy is used to calculate the loss.
However, the number of epochs can be increased to decrease training loss but our objective was to reduce overgeneralization so that model could work well on real-world examples.
Training loss was reduced to 2.6-2.8 range and validation loss being 2.9-3.3 range thus leading to a model with low variance thus low overfitting. In the inference section, we will also give examples of image captioning over different data and illustrate how low variance in our model leads to better performance.

## Scope of Further Improvements
* **Using a larger dataset**: The fundamental scope of every ML model is using larger dataset. We were unable to use larger dataset due to less computational resources but using larger dataset like Flickr30k or MS-COCO will surely make the model better.
* **Changing the model architecture**: One major change we can state is using the new Attention model[Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044).Here we can precisely extract out striking features of an image without getting disturbed by background objects much. But this model has its disadvantages too as we mentioned in our “Inference” Section.
* **Doing more hyperparameter tuning (learning rate, batch size, number of layers, number of units, dropout rate, batch normalization etc.)**: We tried to do a thorough hyperparameter search and provided some of the best hyperparameters combinations with graphs and tables in this report. However, there is still scope for increasing LSTM units and LSTM layers to produce a better model
* Writing the code in a proper object-oriented way so that it becomes easier for others to replicate.
* **Better evaluation metrics**: BLEU-4 score does not evaluate the model as compared to human evaluation. Thus we need to develop better Evaluation Metrics.

## References
1. Oriol Vinyals, Alexander Toshev, Samy Bengio, and Dumitru Erhan. Show and tell: A neural image caption generator. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3156–3164,2015.
1. Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhudinov, Rich Zemel, and Yoshua Bengio. Show, attend and tell Neural image caption generation with visual attention. In Francis Bach and David Blei, editors, Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pages 2048–2057, Lille, France, 07–09 Jul 2015. PMLR.
1. Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, Ping Tak Peter Tang. On Large-batch Training For Deep Learning: Generalization Gap And Sharp Minima. Published as a conference paper at ICLR,  15 Sep 2016.
1. M. Hodosh, P. Young and J. Hockenmaier (2013) "Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics", Journal of Artificial Intelligence Research, Volume 47, pages 853-899, http://www.jair.org/papers/paper3994.html
