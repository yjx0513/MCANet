# -*- coding:utf-8 -*-
'''
Author: MrZQAQ
Date: 2022-03-29 14:00
LastEditTime: 2022-11-23 15:32
LastEditors: MrZQAQ
Description: Test model
FilePath: /MCANet/utils/TestModel.py
'''

import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from transformers import AutoModel, AutoTokenizer


DEVICE = torch.device("cuda")


#local_directory = "/home/4paradigm/MMDG-DTI-main/MMDG-DTI/pretrain/pretrain/local_pubchem_smiles"

# 加载本地分词器
#chem_tokenizer = AutoTokenizer.from_pretrained(local_directory)
#chem_model = AutoModel.from_pretrained(local_directory).to(DEVICE)

chem_tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k", do_lower_case=False)
chem_model = AutoModel.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k").to(DEVICE)



def test_precess(MODEL, pbar, LOSS, DEVICE, FOLD_NUM):
    if isinstance(MODEL, list):
        for item in MODEL:
            item.eval()
    else:
        MODEL.eval()
    test_losses = []
    Y, P, S = [], [], []
    p_LM, d_LM = {}, {}
    with torch.no_grad():
        for i, data in pbar:
            '''data preparation '''
            sequences,smiles,compounds,proteins, labels,protein_LM,molecule_LM,molecule_atoms_new,molecule_adjs_new = data
            compounds = compounds.to(DEVICE)
            proteins = proteins.to(DEVICE)
            labels = labels.to(DEVICE)
            protein_LM = protein_LM.to(DEVICE)
            molecule_LM = molecule_LM.to(DEVICE)
            molecule_atoms_new = molecule_atoms_new.to(DEVICE)
            molecule_adjs_new = molecule_adjs_new.to(DEVICE)
             
            # supports = supports.to(DEVICE)
            
        
            

            if isinstance(MODEL, list):
                predicted_scores = torch.zeros(2).to(DEVICE)
                for i in range(len(MODEL)):
                    predicted_scores = predicted_scores + \
                        MODEL[i](compounds, proteins,protein_LM,molecule_LM,molecule_atoms_new,molecule_adjs_new)[0]
                        # MODEL[i](compounds, proteins,protein_LM,molecule_LM)
                        
                predicted_scores = predicted_scores / FOLD_NUM
            else:
                predicted_scores,_, _, _, _ = MODEL(compounds, proteins,protein_LM,molecule_LM,molecule_atoms_new,molecule_adjs_new) 
            loss = LOSS(predicted_scores, labels)
            correct_labels = labels.to('cpu').data.numpy()
            predicted_scores = F.softmax(
                predicted_scores, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(predicted_scores, axis=1)
            predicted_scores = predicted_scores[:, 1]

            Y.extend(correct_labels)
            P.extend(predicted_labels)
            S.extend(predicted_scores)
            test_losses.append(loss.item())
    Precision = precision_score(Y, P)
    Recall = recall_score(Y, P)
    AUC = roc_auc_score(Y, S)
    tpr, fpr, _ = precision_recall_curve(Y, S)
    PRC = auc(fpr, tpr)
    Accuracy = accuracy_score(Y, P)
    test_loss = np.average(test_losses)
    return Y, P, test_loss, Accuracy, Precision, Recall, AUC, PRC


def test_model(MODEL, dataset_loader, save_path, DATASET, LOSS, DEVICE, dataset_class="Train", save=True, FOLD_NUM=1):
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(dataset_loader)),
        total=len(dataset_loader))
    T, P, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_precess(
        MODEL, test_pbar, LOSS, DEVICE, FOLD_NUM)
    if save:
        if FOLD_NUM == 1:
            filepath = save_path + \
                "/{}_{}_prediction.txt".format(DATASET, dataset_class)
        else:
            filepath = save_path + \
                "/{}_{}_ensemble_prediction.txt".format(DATASET, dataset_class)
        with open(filepath, 'a') as f:
            for i in range(len(T)):
                f.write(str(T[i]) + " " + str(P[i]) + '\n')
    results = '{}: Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};AUC:{:.5f};PRC:{:.5f}.' \
        .format(dataset_class, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test)
    print(results)
    return results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test
