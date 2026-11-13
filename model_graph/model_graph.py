# -*- coding:utf-8 -*-
'''
Author: MrZQAQ
Date: 2022-03-26 19:34
LastEditTime: 2022-11-23 16:34
LastEditors: MrZQAQ
Description: DeepLearing Model
FilePath: /MCANet/model.py
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Informer_block import AttentionLayer, ProbAttention

class EncoderLayer(nn.Module):
    def __init__(self, i_channel, o_channel, growth_rate, groups, pad2=7):
        super(EncoderLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=i_channel, out_channels=o_channel, kernel_size=(2 * pad2 + 1), stride=1,
                               groups=groups, padding=pad2,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(i_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels=o_channel, out_channels=growth_rate, kernel_size=(2 * pad2 + 1), stride=1,
                               groups=groups, padding=pad2,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(o_channel)
        self.drop_rate = 0.1

    def forward(self, x):
        # xn = self.bn1(x)
        xn = self.relu(x)
        xn = self.conv1(xn)
        xn = self.bn2(xn)
        xn = self.relu(xn)
        xn = self.conv2(xn)

        return torch.cat([x, xn], 1)


class Encoder(nn.Module):
    def __init__(self, inc, outc, growth_rate, layers, groups, pad1=15, pad2=7):
        super(Encoder, self).__init__()
        self.layers = layers
        self.relu = nn.ReLU(inplace=True)
        self.conv_in = nn.Conv1d(in_channels=inc, out_channels=inc, kernel_size=(pad1 * 2 + 1), stride=1, padding=pad1,
                                 bias=False)
        self.dense_cnn = nn.ModuleList(
            [EncoderLayer(inc + growth_rate * i_la, inc + (growth_rate // 2) * i_la, growth_rate, groups, pad2) for i_la
             in
             range(layers)])
        self.conv_out = nn.Conv1d(in_channels=inc + growth_rate * layers, out_channels=outc, kernel_size=(pad1 * 2 + 1),
                                  stride=1,
                                  padding=pad1, bias=False)

    def forward(self, x):
        x = self.conv_in(x)
        for i in range(self.layers):
            x = self.dense_cnn[i](x)
        x = self.relu(x)
        x = self.conv_out(x)
        x = self.relu(x)
        return x

class DT_LeNet(nn.Module):
    def __init__(self, hidden, dropout, classes, layers):
        super(DT_LeNet, self).__init__()
        self.CNNs = nn.ModuleList(
            [nn.Conv1d(in_channels=hidden, out_channels=hidden, kernel_size=7, padding=3) for _ in range(layers)])
        self.BN = nn.BatchNorm1d(hidden)  # nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])

        self.FC_combs = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.FC_down = nn.Linear(hidden, 128)
        self.FC_out = nn.Linear(128, classes)
        self.layers = layers
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, dti_feature):
        dti_feature = dti_feature.permute(0, 2, 1)  # self.BN(dti_feature.permute(0, 2, 1))
        for i in range(self.layers):
            dti_feature = self.act(self.CNNs[i](dti_feature)) + dti_feature
        dti_feature = dti_feature.permute(0, 2, 1)
        dti_feature = torch.mean(dti_feature, dim=1)
        GRL_feature = dti_feature.clone()
        for i in range(self.layers):
            dti_feature = self.act(self.FC_combs[i](dti_feature))
        dti_feature = self.FC_down(dti_feature)
        dti = self.FC_out(dti_feature)
        return dti, dti_feature, GRL_feature

class GRL(nn.Module):
    
    def __init__(self, max_iter):
        super(GRL, self).__init__()
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = max_iter

    def forward(self, input):
        self.iter_num += 1
        return input * 1.0

    def backward(self, gradOutput):
        coeff = np.float(2.0 / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - 1)
        return -coeff * gradOutput


class Fusion(nn.Module):
    def __init__(self, hidden1, hidden2, dropout=0.05):
        super(Fusion, self).__init__()
        self.si_L = nn.Sigmoid()
        self.si_S = nn.Sigmoid()
        self.so_f = nn.Sigmoid()
        self.combine = nn.Linear(128 * 4, 128)
        self.ln = nn.LayerNorm(128)
        self.drop = nn.Dropout(p=dropout)


    def forward(self, LM_fea, Sty_fea):

        Sty_fea_norm = Sty_fea * (abs(torch.mean(LM_fea))/abs(torch.mean(Sty_fea)))
        f_h = torch.cat((LM_fea.unsqueeze(1), Sty_fea_norm.unsqueeze(1)), dim=1)

        f_att = torch.mean(f_h, dim=1)
        f_att = self.so_f(f_att)
        fus_fea = torch.cat((LM_fea, Sty_fea, LM_fea * f_att, Sty_fea * f_att), dim=1)
        # print(self.combine(fus_fea))
        fus_fea = self.combine(fus_fea)

        return fus_fea


class Discriminator(nn.Module):
    def __init__(self, max_iter, source_number, device):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Parameter(torch.Tensor(256, 128))#nn.Linear(75, 128)
        self.fc2 = nn.Parameter(torch.Tensor(75, 128))
        self.fc3 = nn.Parameter(torch.Tensor(128, source_number))  # nn.Linear(128, source_number)
        self.drop_lm = nn.Dropout(0.0)
        self.drop_sty = nn.Dropout(0.0)
        self.drop = nn.Dropout(0.5)
        self.grl_layer = GRL(10000)
        self.grl_layer2 = GRL(100)
        self.source_number = source_number
        self.device = device

    def forward(self, feature1, feature2):
        if self.source_number > 2:
            adversarial_out1 = self.grl_layer(self.drop_lm(feature1.detach()))
            adversarial_out2 = self.grl_layer2(self.drop_sty(feature2))
            adversarial_out1 = torch.matmul(adversarial_out1,  nn.init.xavier_uniform_(self.fc1))
            adversarial_out2 = torch.matmul(adversarial_out2, nn.init.xavier_uniform_(self.fc2))
            adversarial_out = adversarial_out2 * torch.sigmoid(adversarial_out1)
            adversarial_out = torch.matmul(self.drop(torch.relu(adversarial_out)), nn.init.xavier_uniform_(self.fc3))
            # print(adversarial_out)
        else:
            adversarial_out = torch.zeros((feature1.shape[0], 1), device=self.device)
        return adversarial_out




class MCANet(nn.Module):
    def __init__(self, hp,
                 protein_MAX_LENGH=1000,
                 drug_MAX_LENGH=100):
        super(MCANet, self).__init__()
        self.dim = hp.char_dim
        self.conv = hp.conv
        self.drug_MAX_LENGTH = drug_MAX_LENGH
        self.drug_kernel = hp.drug_kernel
        self.protein_MAX_LENGTH = protein_MAX_LENGH
        self.protein_kernel = hp.protein_kernel
        self.drug_vocab_size = 65
        self.protein_vocab_size = 26
        self.attention_dim = hp.conv * 4
        self.drug_dim_afterCNNs = self.drug_MAX_LENGTH - \
            self.drug_kernel[0] - self.drug_kernel[1] - self.drug_kernel[2] + 3
        self.protein_dim_afterCNNs = self.protein_MAX_LENGTH - \
            self.protein_kernel[0] - self.protein_kernel[1] - \
            self.protein_kernel[2] + 3
        self.drug_attention_head = 5
        self.protein_attention_head = 7
        self.mix_attention_head = 5

        self.drug_embed = nn.Embedding(
            self.drug_vocab_size, self.dim, padding_idx=0)
        self.protein_embed = nn.Embedding(
            self.protein_vocab_size, self.dim, padding_idx=0)

        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv,
                      kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv*2, out_channels=self.conv * 4,
                      kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )
        self.Drug_max_pool = nn.MaxPool1d(self.drug_dim_afterCNNs)
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv,
                      kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4,
                      kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )
# inc, outc, growth_rate, layers
        self.encoder_protein_LM = Encoder(inc=1024, outc=160, growth_rate=128, layers=3, groups=64, pad1=7, pad2=3)
        self.encoder_drug = Encoder(inc=768, outc=160, growth_rate=128, layers=3, groups=32, pad1=7, pad2=3)
        self.Informer_blocks = nn.ModuleList(
            [AttentionLayer(ProbAttention(None, 3, 0), 160, 1),
             AttentionLayer(ProbAttention(None, 5, 0), 160, 1)])
        
        
        self.Protein_max_pool = nn.MaxPool1d(self.protein_dim_afterCNNs)

        self.mix_attention_layer = nn.MultiheadAttention(
            self.attention_dim, self.mix_attention_head)
        
        self.FC_out1 = DT_LeNet(160, 0.05, 2, 3)

        
    

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.dropout = nn.Dropout(p=0.1)
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.conv*8+128, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)

    def forward(self, drug, protein,protein_LM,molecule_LM):
        # [B, F_O] -> [B, F_O, D_E]
        # [B, T_O] -> [B, T_O, D_E]
        drugembed = self.drug_embed(drug)
        proteinembed = self.protein_embed(protein)
        # [B, F_O, D_E] -> [B, D_E, F_O]
        # [B, T_O, D_E] -> [B, D_E, T_O]
        drugembed = drugembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)

        # [B, D_E, F_O] -> [B, D_C, F_C]
        # [B, D_E, T_O] -> [B, D_C, T_C]
        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)
        
        # [B, D_C, F_C] -> [F_C, B, D_C]
        # [B, D_C, T_C] -> [T_C, B, D_C]
        drug_QKV = drugConv.permute(2, 0, 1)
        protein_QKV = proteinConv.permute(2, 0, 1)

        # cross Attention
        # [F_C, B, D_C] -> [F_C, B, D_C]
        # [T_C, B, D_C] -> [T_C, B, D_C]
        drug_att, _ = self.mix_attention_layer(drug_QKV, protein_QKV, protein_QKV)
        protein_att, _ = self.mix_attention_layer(protein_QKV, drug_QKV, drug_QKV)

        # [F_C, B, D_C] -> [B, D_C, F_C]
        # [T_C, B, D_C] -> [B, D_C, T_C]
        drug_att = drug_att.permute(1, 2, 0)
        protein_att = protein_att.permute(1, 2, 0)


        drugConv = drugConv * 0.5 + drug_att * 0.5
        proteinConv = proteinConv * 0.5 + protein_att * 0.5
        
        
        """DTI 1D feature with pretrain language model"""
        proteins_acids_LM = self.encoder_protein_LM(protein_LM.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1)
        molecule_smiles_LM = self.encoder_drug(molecule_LM.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1)

        DT_1D_Feature = torch.cat((proteins_acids_LM, molecule_smiles_LM), 1)
        DT_1D_P_att = self.dropout(self.Informer_blocks[0](DT_1D_Feature, proteins_acids_LM, proteins_acids_LM, None))
        DT_1D_D_att = self.dropout(self.Informer_blocks[1](DT_1D_Feature, molecule_smiles_LM, molecule_smiles_LM, None))
        DT_1D_Feature = DT_1D_P_att + DT_1D_D_att
        # import pdb;pdb.set_trace()
        dti1d, dti1d_feature, LM_feature = self.FC_out1(DT_1D_Feature)

        
        import pdb;pdb.set_trace()
        
        drugConv = self.Drug_max_pool(drugConv).squeeze(2)
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)

        
        
        # (Pdb) drugConv.shape
        # torch.Size([96, 160])
        # (Pdb) proteinConv.shape
        # torch.Size([96, 160])
        # (Pdb) dti1d_feature.shape
        # torch.Size([96, 128])
        
        

        pair = torch.cat([drugConv, proteinConv, dti1d_feature], dim=1) #torch.Size([96, 448])
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        
        return predict


class onlyPolyLoss(nn.Module):
    def __init__(self, hp,
                 protein_MAX_LENGH=1000,
                 drug_MAX_LENGH=100):
        super(onlyPolyLoss, self).__init__()
        self.dim = hp.char_dim
        self.conv = hp.conv
        self.drug_MAX_LENGH = drug_MAX_LENGH
        self.drug_kernel = hp.drug_kernel
        self.protein_MAX_LENGH = protein_MAX_LENGH
        self.protein_kernel = hp.protein_kernel
        self.drug_vocab_size = 65
        self.protein_vocab_size = 26
        self.attention_dim = hp.conv * 4
        self.durg_dim_afterCNNs = self.drug_MAX_LENGH - \
            self.drug_kernel[0] - self.drug_kernel[1] - self.drug_kernel[2] + 3
        self.protein_dim_afterCNNs = self.protein_MAX_LENGH - \
            self.protein_kernel[0] - self.protein_kernel[1] - \
            self.protein_kernel[2] + 3

        self.drug_embed = nn.Embedding(
            self.drug_vocab_size, self.dim, padding_idx=0)
        self.protein_embed = nn.Embedding(
            self.protein_vocab_size, self.dim, padding_idx=0)

        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv,
                      kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv*2, out_channels=self.conv * 4,
                      kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )
        self.Drug_max_pool = nn.MaxPool1d(self.durg_dim_afterCNNs)
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv,
                      kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4,
                      kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )
        
        self.Protein_max_pool = nn.MaxPool1d(self.protein_dim_afterCNNs)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.conv*8+128, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)

    def forward(self, drug, protein):
        # [B, F_O] -> [B, F_O, D_E]
        # [B, T_O] -> [B, T_O, D_E]
        drugembed = self.drug_embed(drug)
        proteinembed = self.protein_embed(protein)

        # [B, F_O, D_E] -> [B, D_E, F_O]
        # [B, T_O, D_E] -> [B, D_E, T_O]
        drugembed = drugembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)

        # [B, D_E, F_O] -> [B, D_C, F_C]
        # [B, D_E, T_O] -> [B, D_C, T_C]
        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)

        drugConv = self.Drug_max_pool(drugConv).squeeze(2)
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)

        pair = torch.cat([drugConv, proteinConv], dim=1)
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        return predict