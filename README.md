# Generate_TV_Scripts
Udacity Deep Learning Nanodegree Project #3.

* This repo is about how to generate your own TV Scripts using Recurrent Neural Networks.
* It is implemented by using PyTorch library.
* You can refer to Original Udacity repo [here](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-tv-script-generation)

# Project Synopsis

Using the partially provided Seinfield dataset along with Deep Neural Network Algorithms we can generate a new TV Script though that may not exactly look like the Original Script.

# Project Objective

The objective of this Project is to Generate our own Seinfeld TV scripts using RNNs.The input data given is a Seinfeld dataset of scripts from 9 seasons.The Neural Network that we will build will generate a new ,"fake" TV script, based on patterns it recognizes in the given training data.

# Table Of Contents

* Introduction
* Data Description
* Data Pre-Processing
* Building The Neural Network
* Training The Neural Network
* Model Results

# Introduction

Writing scripts for a TV program is a mundane task.Imagine writing every word which comes out from every character in a movie or a tv drama inorder to generate a complete script.Instead of writing manually,here we are going to generate a TV script using Long Short Term Memory Networks – usually just called “LSTMs” – which are a special kind of RNN,capable of learning long-term dependencies.As These networks are capable of predicting and generating the next words that we are intended to generate sequentially in the script.We will train these networks by feeding them an existing TV script. Once the networks are trained we will generate a new TV script.

## LSTMs
I am using LSTMs networks to generate a TV script, which is a special type of Recurrent neural network. Our training data is a small subset of Seinfield Dataset.Seinfield is a TV series having 9 seasons. The dataset contains the conversation between various characters.In this case,I used Recurrent Neural Network(RNN) implemented in PyTorch.

# Data Description

Here we are provided with a small subset of Seinfield Dataset. Seinfield is a TV series having 9 seasons.The dataset contains the conversation between various characters.

# Data Pre-Processing
## Implemented the following Pre-processing Functions on the Dataset

### Lookup Table
* Created a Lookup Table with two dictionaries (Word to ID and ID to Word) used for word embeddings
### Tokenize Punctuation
* Splitted the scripts into word arrays and implemented a function for tokenizing punctuation.The punctuation becomes like another word in the word array. This makes it easier     for the RNN to predict the next word.

# Building The Neural Network
## Implemented the following functions as core components for building the RNN

### Batching The Data
* Implemented the batch_data function to batch words data into chunks of size batch_size using the TensorDataset and DataLoader classes.
### Creating Data Loaders
* Created Data Loaders after creating feature_tensors and target_tensors of the correct size and content for a given sequence_length.
### Initializing RNN Model and Defining Layers
* Implemented an RNN using PyTorch's [Module Class](https://pytorch.org/docs/master/nn.html#torch.nn.Module).Here we may choose to use a GRU or an LSTM.To complete the RNN,       we will have to implement the following functions for the class:
#### __init__ - The initialize function.
* The initialize function will create the layers of the neural network and will save them to the class.
#### init_hidden - The initialization function for an LSTM/GRU hidden state
#### forward - Forward propagation function.
* The forward propagation function will use these layers to run forward propagation and generate an output and a hidden state.
### Forward And Back Propogation
* Here we use the RNN class that was implemented to apply forward and back propagation.This function will be called,iteratively,in the training loop and it should return the       average loss over a batch and the hidden state.
### Initialize The Hidden State Of An LSTM/GRU
*





In this project, we are going to generate a TV script using LSTM Network. We will train these networks by feeding them an existing TV script. Once the networks are trained we will generate a new TV script.s



Language is in many ways the seed in intelligence. We as humans use language to convey informations between each other. But unfortunately, computers can't really understand human languages. New breatktrough in AI has led to more technolgies in language understanding called Natural Language Processing (NLP).


Language communication can be done in many ways.We as humans use language to convey informations between each other.But unfortunately,computers can't really understand human languages.New breakthrough in AI has led to more technolgies in language understanding called Natural Language Processing (NLP).

# Natural Language Processing

NLP is the study that focuses on the interactions between human language and computers. By using NLP, developers can organize and structure knowledge to perform tasks such as automatic summarization, translation, named entity recognition, relationship extraction, sentiment analysis, speech recognition, and topic segmentation.


# Project Objective
Generate your own Seinfeld TV scripts using RNNs.The input data was Seinfeld dataset of scripts from 9 seasons.The Neural Network you'll build will generate a new ,"fake" TV script, based on patterns it recognizes in this training data.
