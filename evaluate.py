import argparse
import os
import sys
import h5py as h5
import numpy as np
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def trans__logits_array(seq):
    '''
    transform logits to label: -1, 0, 1, ....
    '''
    seq[:] = [x + 1 for x in seq]
    for i in range(len(seq)):
        if(seq[i]== 17):
            seq[i] = 0
    return seq


def trans_labels_array(seq):
    seq[:] = [x + 1 for x in seq]
    return seq


def get_binary(seq):
    for i in range(len(seq)):
        if(seq[i] > 0):
            seq[i] = 1
    return seq



def ji_score(seq_true, seq_pred):
    return jaccard_similarity_score(seq_true, seq_pred)


def read_logits(file_path, h5_path):
    with h5.File(file_path) as f:
        label_index = list(f["label_index"])
        grp = f[h5_path]
        timestamps = np.array(grp["timestamps"])
        labels = np.array(grp["labels"])
        logits = np.array(grp["data"])
        
        return label_index, timestamps, labels, logits

def read_gru(file_path, h5_path):
    with h5.File(file_path) as f:
        label_index = list(f["label_index"])
        grp = f[h5_path]
        timestamps = np.array(grp["timestamps"])
        labels = np.array(grp["labels"])
        data= np.array(grp["data"])
        return label_index, timestamps, labels, data

def get_accuracy(seq_true, seq_pred):
    count = 0
    for i in range(len(seq_pred)):
        if(seq_true[i] == seq_pred[i]):
            count = count + 1
    accuracy = count / len(seq_pred)
    return accuracy


def get_precision(seq_true, seq_pred, model):
    if(model == 'macro'):
        return precision_score(seq_true, seq_pred, average='macro')
    elif(model == 'weighted'):
        return precision_score(seq_true, seq_pred, average='macro')
    elif(model == 'samples'):
        return precision_score(seq_true, seq_pred, average='samples')
    elif(model == None):
        return precision_score(seq_true, seq_pred, average= None)
    else:
        return precision_score(seq_true, seq_pred, average='micro')


def get_f1_score(seq_true, seq_pred, model):
    if(model == 'macro'):
        return f1_score(seq_true, seq_pred, average='macro')
    elif(model == 'weighted'):
        return f1_score(seq_true, seq_pred, average='macro')
    elif(model == None):
        return f1_score(seq_true, seq_pred, average=None)
    else:
        return f1_score(seq_true, seq_pred, average='micro')


def test_y():
    y_true = [0, 1, 2, 0, 1, 2, 1, 0, 0, 2, 2, 0, 0]
    y_pred = [0, 2, 1, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0]
    b_true = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]
    b_pred = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]

    ji_test = jaccard_similarity_score(y_true, y_pred)
    print("the ji score of test is {}".format(ji_test))
    print("the precision of test is {}".format(precision_score(y_true, y_pred, average='micro')))
    print("the precision of test None is {}".format(precision_score(y_true, y_pred, average=None)))
    print("the recall of test is {}".format(recall_score(y_true, y_pred, average='micro')))
    print("the accuracy of test is {}".format(accuracy_score(y_true, y_pred)))
    print("the multi-f1 of test is {}".format(f1_score(y_true, y_pred, average=None)))
    print("the f1 of test is {}".format(f1_score(y_true, y_pred, average='micro')))

    j1_2 = jaccard_similarity_score(b_true, b_pred)
    print("the ji score of b is {}".format(j1_2))
    print("the precision of b is {}".format(precision_score(b_true, b_pred, average='micro')))
    print("the precision of b None is {}".format(precision_score(b_true, b_pred, average=None)))
    print("the recall of b is {}".format(recall_score(b_true, b_pred, average='micro')))
    print("the accuracy of b is {}".format(accuracy_score(b_true, b_pred)))
    print("the f1 of b None is {}".format(f1_score(b_true, b_pred, average=None)))

    cm = confusion_matrix(y_true, y_pred)
    recall2 = np.diag(cm) / np.sum(cm, axis=1)
    precision2 = np.diag(cm) / np.sum(cm, axis=0)
    print(recall2)
    print(precision2)


def print_result(labels, logits):
    index = np.argmax(logits, axis=1)
    logits_index = trans__logits_array(index)
    labels = trans_labels_array(labels)
    print("the set of logits_index is {}".format(set(logits_index)))
    print("the set of labels is {}".format(set(labels)))
    ji_score_1 = ji_score(labels, logits_index)
    print("the global ji_score is {}".format(ji_score_1))
    accuracy = get_accuracy(labels, logits_index)
    print("the accuracy is {}".format(accuracy))
    individual_precision= get_precision(labels, logits_index, None)
    # print("the individual precision is {}".format(individual_precision))
    individual_f1_score = get_f1_score(labels, logits_index, None)
    global_f1_score = get_f1_score(labels, logits_index,'micro')
    print("the individual f1_score is {}".format(individual_f1_score))
    print("the global f1_score is {}".format(global_f1_score))
    return individual_f1_score


def print_binary_result(labels, logits):
    index = np.argmax(logits, axis=1)
    logits_index = trans__logits_array(index)
    labels = trans_labels_array(labels)

    binary_logits = get_binary(logits_index)
    binary_labels = get_binary(labels)

    binary_f1_score = f1_score(binary_labels, binary_logits, average='micro')
    binary_f1_score_each = f1_score(binary_labels, binary_logits, average=None)
    print("the f1_score of binary is {}".format(binary_f1_score))
    print("the f1_score of binary of each is {}".format(binary_f1_score_each))




label_index, timestamps, labels, data = read_logits("../result_from_server/gru-events.h5", "/classifications/marten-20")

label_index_b, timestamps_b, labels_b, data_b = read_logits("../result_from_server/gru-events.h5", "/classifications/marten-20")




print("--------- RNN------------")
print_result(labels, data)
print_binary_result(labels_b, data_b)


# print("--------- Dec------------")
# print_result(d_labels, d_logits)
# print_binary_result(d_labels_b, d_logits_b)
#
#
# print("--------- Seg------------")
# print_result(labels, logits)
# print_binary_result(labels_b, logits_b)



