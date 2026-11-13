import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from torch.utils.data import DataLoader
from tqdm import tqdm

from config import hyperparameter
from model_graph.model_lmm_dis import MCANet
# from model_ori import MCANet
from utils.DataPrepare import get_kfold_data, shuffle_dataset
# from utils.DataSetsFunction import CustomDataSet, collate_fn
from utils.DataSets_graph import CustomDataSet, collate_fn
from utils.EarlyStoping import EarlyStopping
from LossFunction import CELoss, PolyLoss
from utils.TestModel import test_model
from utils.ShowResult import show_result
from transformers import AutoModel, AutoTokenizer


DEVICE = torch.device("cuda")





def run_model(SEED, DATASET, MODEL, K_Fold, LOSS):

    '''set random seed'''
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    '''init hyperparameters'''
    hp = hyperparameter()

    '''load dataset from text file'''
    assert DATASET in ["DrugBank", "KIBA", "Davis", "Enzyme", "GPCRs", "ion_channel"]
    print("Train in " + DATASET)
    print("load data")
    dir_input = ('./DataSets/{}.txt'.format(DATASET))
    with open(dir_input, "r") as f:
        data_list = f.read().strip().split('\n')
    print("load finished")

    '''set loss function weight'''
    if DATASET == "Davis":
        weight_loss = torch.FloatTensor([0.3, 0.7]).to(DEVICE)
    elif DATASET == "KIBA":
        weight_loss = torch.FloatTensor([0.2, 0.8]).to(DEVICE)
    else:
        weight_loss = None

    '''shuffle data'''
    print("data shuffle")
    data_list = shuffle_dataset(data_list, SEED)

    '''split dataset to train&validation set and test set'''
    split_pos = len(data_list) - int(len(data_list) * 0.2)
    train_data_list = data_list[0:split_pos]
    test_data_list = data_list[split_pos:-1]
    print('Number of Train&Val set: {}'.format(len(train_data_list)))
    print('Number of Test set: {}'.format(len(test_data_list)))

    '''metrics'''
    Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable, Precision_List_stable = [], [], [], [], []

    for i_fold in range(K_Fold):
        print('*' * 25, 'No.', i_fold + 1, '-fold', '*' * 25)
        fold_size = int(len(train_data_list) *0.8)
        # import pdb;pdb.set_trace()
        # valid_dataset = train_data_list[fold_size:] 
        # train_dataset = train_data_list[0:fold_size]

        train_dataset, valid_dataset = get_kfold_data(
            i_fold, train_data_list, k=K_Fold)
        train_dataset = CustomDataSet(train_dataset)
        valid_dataset = CustomDataSet(valid_dataset)
        test_dataset = CustomDataSet(test_data_list)
        train_size = len(train_dataset)
        print(len(train_dataset))
        # import pdb;pdb.set_trace()

        train_dataset_loader = DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, num_workers=4,
                                          collate_fn=collate_fn, drop_last=True)
        valid_dataset_loader = DataLoader(valid_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=4,
                                          collate_fn=collate_fn, drop_last=True)
        test_dataset_loader = DataLoader(test_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=4,
                                         collate_fn=collate_fn, drop_last=True)

        """ create model"""
        model = MODEL(hp).to(DEVICE)

        """Initialize weights"""
        '''权重参数 偏置参数'''
        weight_p, bias_p = [], []
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]

        """create optimizer and scheduler"""
        optimizer = optim.AdamW(
            [{'params': weight_p, 'weight_decay': hp.weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=hp.Learning_rate)
        # optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate*10, cycle_momentum=False,
                                                step_size_up=train_size // hp.Batch_size)

        '''Loss Function'''
        if LOSS == 'PolyLoss':
            Loss = PolyLoss(weight_loss=weight_loss,
                            DEVICE=DEVICE, epsilon=hp.loss_epsilon)
        else:
            Loss = CELoss(weight_CE=weight_loss, DEVICE=DEVICE)

        """Output files"""
        save_path = "./" + DATASET +"/LLM"+ "/{}".format(i_fold+1) +"_base"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_results = save_path + '/' + 'The_results_of_whole_dataset_fpnadd.txt'

        early_stopping = EarlyStopping(
            savepath=save_path, patience=hp.Patience, verbose=True, delta=0)

        """Start training."""
        print('Training...')
        p_LM, d_LM = {}, {}
        for epoch in range(1, hp.Epoch + 1):
            if early_stopping.early_stop == True:
                break
            train_pbar = tqdm(
                enumerate(BackgroundGenerator(train_dataset_loader)),
                total=len(train_dataset_loader))

            """train"""
            train_losses_in_epoch = []
            model.train()
            for train_i, train_data in train_pbar:
                
                sequences,smiles,train_compounds, train_proteins, train_labels,protein_LM,molecule_LM,molecule_atoms_new,molecule_adjs_new = train_data
                # sequences,smiles,compound_new, d_LM,protein_new, labels_new
                # train_compounds,atom_features, adjs, train_proteins, train_labels = train_data
                train_compounds = train_compounds.to(DEVICE)
                # edges = edges.to(DEVICE)
                train_proteins = train_proteins.to(DEVICE)
                train_labels = train_labels.to(DEVICE)
                protein_LM = protein_LM.to(DEVICE)
                molecule_LM = molecule_LM.to(DEVICE)
                molecule_atoms_new = molecule_atoms_new.to(DEVICE)
                molecule_adjs_new = molecule_adjs_new.to(DEVICE)
             
                optimizer.zero_grad()
                # import pdb;pdb.set_trace()
                # predicted_interaction = model(train_compounds, train_proteins, edges, supports)
                predict,dti1d, dti2d, dti1d_feature, dti2d_feature = model(train_compounds, train_proteins,protein_LM,molecule_LM,molecule_atoms_new,molecule_adjs_new)


                # predicted_interaction = model(train_compounds, train_proteins,supports)
                
                train_loss = Loss(predict,train_labels)
                # loss1 = Loss(predict_old,train_labels)
                loss2 = F.cross_entropy(dti1d, train_labels)
                loss3 = F.cross_entropy(dti2d, train_labels)
                train_loss = train_loss + loss2 + 1.25 * loss3 
                # loss_guide = 2 * torch.exp(model.param)**2
                
                # train_loss = train_loss - 0.01 *torch.mean(loss_guide)
                train_losses_in_epoch.append(train_loss.item())
                train_loss.backward()
                optimizer.step()
                scheduler.step()
            train_loss_a_epoch = np.average(
                train_losses_in_epoch)  # 一次epoch的平均训练loss

            """valid"""
            valid_pbar = tqdm(
                enumerate(BackgroundGenerator(valid_dataset_loader)),
                total=len(valid_dataset_loader))
            valid_losses_in_epoch = []
            model.eval()
            Y, P, S = [], [], []
            with torch.no_grad():
                for valid_i, valid_data in valid_pbar:
                    # sequences,smiles,train_compounds,edges,supports, train_proteins, train_labels

                    sequences,smiles,valid_compounds,valid_proteins, valid_labels,protein_LM,molecule_LM,molecule_atoms_new,molecule_adjs_new = valid_data

                    valid_compounds = valid_compounds.to(DEVICE)
                    valid_proteins = valid_proteins.to(DEVICE)
                    valid_labels = valid_labels.to(DEVICE)
                    protein_LM = protein_LM.to(DEVICE)
                    molecule_LM = molecule_LM.to(DEVICE)
                    molecule_atoms_new = molecule_atoms_new.to(DEVICE)
                    molecule_adjs_new = molecule_adjs_new.to(DEVICE)
             

                    valid_scores ,dti1d, dti2d, dti1d_feature, dti2d_feature= model(valid_compounds, valid_proteins,protein_LM,molecule_LM,molecule_atoms_new,molecule_adjs_new)
                    valid_loss = Loss(valid_scores,valid_labels)
                    valid_losses_in_epoch.append(valid_loss.item())
                    valid_labels = valid_labels.to('cpu').data.numpy()
                    valid_scores = F.softmax(
                        valid_scores, 1).to('cpu').data.numpy()
                    valid_predictions = np.argmax(valid_scores, axis=1)
                    valid_scores = valid_scores[:, 1]

                    Y.extend(valid_labels)
                    P.extend(valid_predictions)
                    S.extend(valid_scores)

            Precision_dev = precision_score(Y, P)
            Reacll_dev = recall_score(Y, P)
            Accuracy_dev = accuracy_score(Y, P)
            AUC_dev = roc_auc_score(Y, S)
            tpr, fpr, _ = precision_recall_curve(Y, S)
            PRC_dev = auc(fpr, tpr)
            valid_loss_a_epoch = np.average(valid_losses_in_epoch)

            epoch_len = len(str(hp.Epoch))
            print_msg = (f'[{epoch:>{epoch_len}}/{hp.Epoch:>{epoch_len}}] ' +
                         f'train_loss: {train_loss_a_epoch:.5f} ' +
                         f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                         f'valid_AUC: {AUC_dev:.5f} ' +
                         f'valid_PRC: {PRC_dev:.5f} ' +
                         f'valid_Accuracy: {Accuracy_dev:.5f} ' +
                         f'valid_Precision: {Precision_dev:.5f} ' +
                         f'valid_Reacll: {Reacll_dev:.5f} ')
            print(print_msg)

            '''save checkpoint and make decision when early stop'''
            early_stopping(Accuracy_dev, model, epoch)

        '''load best checkpoint'''
        model.load_state_dict(torch.load(
            early_stopping.savepath + '/valid_best_checkpoint.pth'))

        '''test model'''
        trainset_test_stable_results, _, _, _, _, _ = test_model(
            model, train_dataset_loader, save_path, DATASET, Loss, DEVICE, dataset_class="Train", FOLD_NUM=1)
        validset_test_stable_results, _, _, _, _, _ = test_model(
            model, valid_dataset_loader, save_path, DATASET, Loss, DEVICE, dataset_class="Valid", FOLD_NUM=1)
        testset_test_stable_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_model(
            model, test_dataset_loader, save_path, DATASET, Loss, DEVICE, dataset_class="Test", FOLD_NUM=1)
        AUC_List_stable.append(AUC_test)
        Accuracy_List_stable.append(Accuracy_test)
        AUPR_List_stable.append(PRC_test)
        Recall_List_stable.append(Recall_test)
        Precision_List_stable.append(Precision_test)
        with open(save_path + '/' + "The_results_of_whole_dataset.txt", 'a') as f:
            f.write("Test the stable model" + '\n')
            f.write(trainset_test_stable_results + '\n')
            f.write(validset_test_stable_results + '\n')
            f.write(testset_test_stable_results + '\n')

    show_result(DATASET, Accuracy_List_stable, Precision_List_stable,
                Recall_List_stable, AUC_List_stable, AUPR_List_stable, Ensemble=False)


def ensemble_run_model(SEED, DATASET, K_Fold):

    '''set random seed'''
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    '''init hyperparameters'''
    hp = hyperparameter()

    '''load dataset from text file'''
    assert DATASET in ["DrugBank", "KIBA", "Davis"]
    print("Train in " + DATASET)
    print("load data")
    dir_input = ('./DataSets/{}.txt'.format(DATASET))
    with open(dir_input, "r") as f:
        data_list = f.read().strip().split('\n')
    print("load finished")

    '''set loss function weight'''
    if DATASET == "Davis":
        weight_loss = torch.FloatTensor([0.3, 0.7]).to(DEVICE)
    elif DATASET == "KIBA":
        weight_loss = torch.FloatTensor([0.2, 0.8]).to(DEVICE)
    else:
        weight_loss = None

    '''shuffle data'''
    print("data shuffle")
    data_list = shuffle_dataset(data_list, SEED)

    '''split dataset to train&validation set and test set'''
    split_pos = len(data_list) - int(len(data_list) * 0.2)
    test_data_list = data_list[split_pos:-1]
    print('Number of Test set: {}'.format(len(test_data_list)))

    save_path = f"./{DATASET}/ensemble"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    test_dataset = CustomDataSet(test_data_list)
    test_dataset_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                                     collate_fn=collate_fn, drop_last=True)

    model = []
    for i in range(K_Fold):
        model.append(MCANet(hp).to(DEVICE))
        '''MCANet K-Fold train process is necessary'''
        try:
            model[i].load_state_dict(torch.load(
                f'./{DATASET}/new/{i+1}' + 'att'+'/valid_best_checkpoint.pth', map_location=torch.device(DEVICE)))
        except FileNotFoundError as e:
            print('-'* 25 + 'ERROR' + '-'*25)
            error_msg = 'Load pretrained model error: \n' + \
                        str(e) + \
                        '\n' + 'MCANet K-Fold train process is necessary'
            print(error_msg)
            print('-'* 55)
            exit(1)

    Loss = PolyLoss(weight_loss=weight_loss,
                    DEVICE=DEVICE, epsilon=hp.loss_epsilon)

    testdataset_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_model(
        model, test_dataset_loader, save_path, DATASET, Loss, DEVICE, dataset_class="Test", save=True, FOLD_NUM=K_Fold)

    show_result(DATASET, Accuracy_test, Precision_test,
                Recall_test, AUC_test, PRC_test, Ensemble=True)