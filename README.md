#  Neuro ConGD

![dvs-record](E:\git\FLGR\doc\dvs-record.gif)

Neuro ConGD dataset is a __neuromorphic continuous gesture dataset__, which comprises 2140 instances of a set of 17 gestures recorded in random order. The gestures include beckoning, finger-snap, ok, push-hand(down, left, right, up), rotate-outward, swipe(left, right, up), tap-index, thumbs-up, zoom(in, out) 

![dataset_overview](E:\git\FLGR\doc\dataset_overview.png)

We design a RNN classifier as a __baseline__ which can be seen in `train_rnn_classifier.py`.

 We hope the Neuro ConGD will become useful resource for the __neuromorphic vision research community__.

_Created by Guang Chen, Jieneng Chen, Marten Lienen, Joerg Concadt, Florian Roehrbein and Alois Knoll_

## News!
__Neuromorphic Continuous Gesture Data (prepared & raw & label) are released__.

___We are going to enlarge our dataset. Stay tuned!___

## Contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Quick Start](#quick-start)
4. [Evaluation metrics](#evaluate)
5. [Funding](#funding)
6. [License](#license)

## Introduction
This work is based on our [research paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Chen_LiDAR-Video_Driving_Dataset_CVPR_2018_paper.html), which appears in CVPR 2018. We propose a large-scale dataset for driving behavior learning, namely, DBNet. You can also check our [dataset webpage](http://www.dbehavior.net/) for a deeper introduction.

In this repository, we release __demo code__ and __partial prepared data__ for training with only images, as well as leveraging feature maps or point clouds. The prepared data are accessible [here](https://drive.google.com/open?id=14RPdVTwBTuCTo0tFeYmL_SyN8fD0g6Hc). (__More demo models and scripts are released soon!__)

## Requirements

* **Tensorflow 1.2.0rc2**
* Python 3.6
* CUDA 8.0+ (For GPU)
* Python Libraries: numpy, scipy, pandas, Theano and scikit-learn

The code has been tested with Python 3.6, Tensorflow 1.2.0rc2 on Ubuntu 16.04 and Win10. But it may work on more machines (directly or through mini-modification).

## Quick Start

### Preprocessing
The users can preprocess the data according to their demand:

    python preprocess_events.py [-h] [dirs [dirs ...]] out

The names of the models are consistent with our [paper](http://www.dbehavior.net/publications.html).



### Training
To train a __baseline__ model to predict gesture class in each timestamp:

    python train_rnn_classifier.py [-h] [--log-dir LOG_DIR] [--learning-rate LEARNING_RATE] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--layers LAYERS] [--validation VALIDATION] dataset

Log files and network parameters will be saved to `logs` folder in default.

To see HELP for the training script:

    python train_rnn_classifier -h

We can use TensorBoard to view the network architecture and monitor the training progress.

    tensorboard --logdir logs

### Evaluation    
After training, you could evaluate the performance of models using `evaluate.py`. 

### Prediction
To get the result of predictions of test data with the RNN baseline:

    python pred_rnn_classifier.py log_dir dataset out

To see HELP for the training script:

    python pred_rnn_classifier -h

The results are saved in `out`  defined by users.

The result directory will be created automatically if it doesn't exist.

## Evaluation metrics
To measure the performance of continuous gesture recognition, we use __mean Jaccard Index__. We also utilize __F-score__ for measuring the performance of gesture detection.

The implementations of these metrics could be found in `evaluate.py`.

## Funding
The building of Neuro ConGD dataset has received funding from the European Union’s Horizon 2020 Research and Innovation Program under Grant Agreement No. 785907 (HBP SGA2).


## License
Our code is released under MIT License. 