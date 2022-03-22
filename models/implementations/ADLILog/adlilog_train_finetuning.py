from model import *
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
from tqdm import tqdm
# from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler

import json

from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import precision_recall_curve, average_precision_score

torch.random.seed = 0
np.random.seed(0)

from datasets import *
from tokenizer import LogTokenizer
from utils import get_padded_data
# from model.loss_function_hyperplane import SimpleLossCompute
from model.model import LogModel, ADLILog
from model.trainer import run_train, run_test
# USE THIS CODE FOR adlilog HYPERPLANE

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from adlilog import adlilogRunner
import pandas as pd
from collections import defaultdict
# USE THIS CODE FOR adlilog HYPERSPHERICAL
import argparse


# parser = argparse.ArgumentParser(description='Run LogClass')
# parser.add_argument('datasetname', metavar='datasetname', type=str, help='the name of the dataset')
# args = parser.parse_args()

parsed_value = "BGL"
# parsed_value = args.datasetname

local_paths = {
    "BGL": "BGL/bgl_0.8/",
    "HDFS": "HDFS/hdfs_0.8/",
}

dataset_name = local_paths[parsed_value]
datasets = {
    "BGL":{
        "training_datasetpath_raw_msgs": "../deep-loglizer/data/processed/"+dataset_name+"train_raw.csv",
        "training_datasetpath_labels": "../deep-loglizer/data/processed/"+dataset_name+"train_labels.csv",
        "test_datasetpath_raw_msgs": "../deep-loglizer/data/processed/"+dataset_name+"test_raw.csv",
        "test_datasetpath_labels": "../deep-loglizer/data/processed/"+dataset_name+"test_labels.csv",
        "store_predictions_gmean": "../deep-loglizer/data/processed/"+dataset_name+"ADLILog_predictions_gmean.csv",
        "store_predictions_fscore": "../deep-loglizer/data/processed/"+dataset_name+"ADLILog_predictions_fscore.csv",
        "store_results_gmean": "../deep-loglizer/data/processed/"+dataset_name+"ADLILog_results_gmean.json",
        "store_results_fscore": "../deep-loglizer/data/processed/"+dataset_name+"ADLILog_results_fscore.json",
        "auxiliary_data": "../../../data/auxiliary_labels/anomalies_github.csv",
    },
    "HDFS": {
        "training_datasetpath_raw_msgs": "../deep-loglizer/data/processed/" + dataset_name + "train_raw.csv",
        "training_datasetpath_labels": "../deep-loglizer/data/processed/" + dataset_name + "train_labels.csv",
        "test_datasetpath_raw_msgs": "../deep-loglizer/data/processed/" + dataset_name + "test_raw.csv",
        "test_datasetpath_labels": "../deep-loglizer/data/processed/" + dataset_name + "test_labels.csv",
        "store_predictions_gmean": "../deep-loglizer/data/processed/" + dataset_name + "ADLILog_predictions_gmean.csv",
        "store_predictions_fscore": "../deep-loglizer/data/processed/" + dataset_name + "ADLILog_predictions_fscore.csv",
        "store_results_gmean": "../deep-loglizer/data/processed/" + dataset_name + "ADLILog_results_gmean.json",
        "store_results_fscore": "../deep-loglizer/data/processed/" + dataset_name + "ADLILog_results_fscore.json",
        "auxiliary_data": "../../../data/auxiliary_labels/anomalies_github.csv",
    },
}

model_params = {
    "tgt_vocab": 2,
    "n_layers": 2,
    "in_features": 16,
    "out_features": 16,
    "num_heads": 2,
    "dropout": 0.05,
    "max_len": 50
}

batch_size = 2048
epochs = 100
pad_len = 50
seed = 0
pretrain_model_path = "./stored_model/pretrained_model.pth"
pretrain_tokenizer_path = "./stored_model/tokenizer.pkl"


def tokenize_data(tokenizer, log_payload, labels):
    dataset_size = log_payload.shape[0]
    data_tokenized = []
    remove_index = []
    lab = []
    for i in range(0, dataset_size):
        tokenized = tokenizer.tokenize(log_payload[i])
        if tokenized is not None:
            data_tokenized.append(tokenized)
            lab.append(labels[i])
        else:
            remove_index.append(i)
    return tokenizer, data_tokenized, np.array(lab).ravel()

def get_padded_data(data, pad_len):
    pd = pad_sequences(data, maxlen=pad_len, dtype="long", truncating="post", padding="post")
    return pd


def create_data_loaders(load_train, labels_train, load_test, labels_test):

    train_data = TensorDataset(torch.tensor(get_padded_data(load_train, pad_len=pad_len), dtype=torch.long),
                               torch.tensor(labels_train.astype(np.int32), dtype=torch.long))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    test_data = TensorDataset(torch.tensor(get_padded_data(load_test, pad_len=pad_len), dtype=torch.long),
                              torch.tensor(labels_test.astype(np.int32).flatten(), dtype=torch.long))
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    return train_dataloader, test_dataloader


def run_optimizer(model, train_dataloader, test_dataloader, labels_test):

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.001)
    distances_ = []
    train_time = []
    test_time = []
    max_auc = 0
    max_auc_epoch = 0

    for epoch in range(epochs):
        model.train()
        print("Epoch", epoch + 1)
        start_time = time.time()
        # train
        run_train(train_dataloader, model,
                  SimpleLossCompute(model, loss_criterion, optimizer, generator=False),
                  step_size=200)
        train_time.append(time.time() - start_time)

        # test
        model.eval()
        start_time = time.time()
        distances = run_test(test_dataloader, model, SimpleLossCompute(model, loss_criterion, None, is_test=True, generator=False), step_size=200)
        test_time.append(time.time() - start_time)


        auc = roc_auc_score(labels_test.astype(np.int32), distances)

        if auc>max_auc:
            max_auc = auc
            max_auc_epoch = epoch


        print("AUC:", auc)
        print(roc_auc_score(labels_test.astype(np.int32), distances))
        distances_.append(np.array(distances).round(3))

    return distances_[max_auc_epoch], labels_test, train_time, test_time

class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, model, criterion, opt=None, is_test=False, generator=False):
        self.model = model
        self.criterion = criterion
        self.opt = opt
        self.is_test = is_test
        self.bce = torch.nn.CrossEntropyLoss()
        self._lambda = 0.05
        self.generator = generator

    def __call__(self, x, y, dist):
        loss = torch.mean((1 - y) * torch.sqrt(dist) - (y) * torch.log(1 - torch.exp(-torch.sqrt(dist))))
        # if self.generator:
        #     out = self.model.generator(x)
        #     loss2 = self.bce(out, y)
        #     loss = loss + (loss2 * self._lambda)

        if not self.is_test:
            loss.backward()
            if self.opt is not None:
                self.opt.step()
                self.opt.zero_grad()

        return loss.item()



def store_results(test_data, test_labels, final_preds, max_distances, store_predictions, store_results):
    print("Confusion Matrix".format(confusion_matrix(test_labels, final_preds)))
    print("F1 score {}".format(f1_score(test_labels, final_preds)))
    print("precision score {}".format(precision_score(test_labels, final_preds)))
    print("recall score {}".format(recall_score(test_labels, final_preds)))
    print("roc_auc score {}".format(roc_auc_score(test_labels, max_distances)))
    print("avg precision score {}".format(average_precision_score(test_labels, max_distances)))

    results = {}

    results["f1_score"] = f1_score(test_labels, final_preds)
    results["precision_score"] = precision_score(test_labels, final_preds)
    results["recall_score"] = recall_score(test_labels, final_preds)
    results["roc_auc_score"] = roc_auc_score(test_labels, max_distances)
    results["avg_prec_score"] = average_precision_score(test_labels, max_distances)
    cf = confusion_matrix(test_labels, final_preds)

    if cf.shape[0] > 1:
        results["fp"] = cf[0, 1]
        results["tp"] = cf[1, 1]
        results["tn"] = cf[0, 0]
        results["fn"] = cf[1, 0]
        results["TNR"] = cf[0, 0] / (cf[0, 0] + cf[1, 0])

    for key in results.keys():
        results[key] = float(results[key])

    print(test_labels.shape)
    print(np.array(final_preds).reshape(-1, 1).shape)

    d = {}
    d["Content"] = test_data
    d["true"] = test_labels
    d["prediction"] = np.array(final_preds).reshape(-1, 1)

    k = np.vstack([test_data.values.ravel(), test_labels.values.ravel(), final_preds.ravel()])
    pom = pd.DataFrame(k).T
    pom.to_csv(store_predictions, index=False)
    del (pom)

    with open(store_results, "w") as file:
        json.dump(results, file)



training_target_data, training_target_labels = pd.read_csv(datasets[parsed_value]["training_datasetpath_raw_msgs"]).fillna("None"), pd.read_csv(datasets[parsed_value]["training_datasetpath_labels"])
training_target_data = training_target_data[training_target_labels.label==0].reset_index().iloc[:, 1:]
training_target_labels = training_target_labels[training_target_labels.label==0].reset_index().iloc[:, 1:]

aux_data = pd.read_csv(datasets[parsed_value]["auxiliary_data"])
aux_data.columns = ["logs"]
anom_labels = pd.DataFrame(np.ones(aux_data.shape[0]))
anom_labels.columns = ["label"]
training_target_data = training_target_data.append(aux_data).reset_index().iloc[:, 1:].values
training_target_labels = training_target_labels.append(anom_labels).reset_index().iloc[:, 1:].values

test_data, test_labels = pd.read_csv(datasets[parsed_value]["test_datasetpath_raw_msgs"]).fillna("None"), pd.read_csv(datasets[parsed_value]["test_datasetpath_labels"])
test_data_size = test_data.shape[0]

tokenizer = LogTokenizer(tokens_file="./stored_model/PreTrainedTokenizer.json")

tokenizer, tokenized_train_msg, tokenized_train_lab = tokenize_data(tokenizer, training_target_data.ravel(), training_target_labels.ravel())
print("size tokenizer pre {}".format(len(tokenizer.word2index)))
tokenizer.update_tokenizer = False
tokenizer1, tokenized_test_msg, tokenized_test_lab = tokenize_data(tokenizer, test_data.values.ravel(), test_labels.values.ravel())
print("size tokenizer pre {}".format(len(tokenizer1.word2index)))


load_train, load_test = get_padded_data(tokenized_train_msg, pad_len=model_params["max_len"]), get_padded_data(tokenized_test_msg, pad_len=model_params["max_len"])

train_dataloader, test_dataloader = create_data_loaders(load_train, tokenized_train_lab, load_test, tokenized_test_lab)


torch.cuda.set_device(0)
un, class_count = np.unique(tokenized_train_lab, return_counts=True)

print(class_count)
calculate_weights = lambda x, i: x.sum() / (len(x) * x[i])
weights = [calculate_weights(class_count, i) for i in range(len(class_count))]
weights /= max(weights)

print(type(weights))
w1 = weights[0]
w2 = weights[1]
weights = [w2, w1]


class_weights = torch.FloatTensor(weights).cuda()

print(class_weights)

loss_criterion = nn.CrossEntropyLoss(weight=class_weights)

src_vocab = len(tokenizer.word2index)

model = LogModel(src_vocab, model_params["tgt_vocab"], n_layers=model_params["n_layers"], in_features=model_params["in_features"],
                   out_features=model_params["out_features"], num_heads=model_params["num_heads"], dropout=model_params["dropout"], max_len=model_params["max_len"], weights_path=pretrain_model_path)

model = model.get_model()
model = ADLILog(model, in_features=model_params["in_features"])
model.to("cuda")
max_distances, labels_test, train_time, test_time = run_optimizer(model, train_dataloader=train_dataloader, test_dataloader=test_dataloader, labels_test=tokenized_test_lab)

fpr, tpr, thresholds = roc_curve(labels_test, max_distances)
index_gmean = np.argmax(np.sqrt(tpr*(1-fpr)))
final_preds_gmean = np.where(max_distances<thresholds[index_gmean], 0, 1)
store_results(test_data, test_labels, final_preds_gmean, max_distances, store_predictions=datasets[parsed_value]["store_predictions_gmean"], store_results=datasets[parsed_value]["store_results_gmean"])

precision, recall, thresholds = precision_recall_curve(labels_test, max_distances)
index_f = np.argmax((2 * precision * recall) / (precision + recall + 0.0001))
final_preds_f1 = np.where(max_distances<thresholds[index_f], 0, 1)

store_results(test_data, test_labels, final_preds_f1, max_distances, store_predictions=datasets[parsed_value]["store_predictions_fscore"], store_results=datasets[parsed_value]["store_results_fscore"])
