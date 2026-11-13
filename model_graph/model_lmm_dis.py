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
from torch.nn import Parameter

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


# class Fusion(nn.Module):
#     def __init__(self, hidden1, hidden2, dropout=0.05):
#         super(Fusion, self).__init__()
#         self.si_L = nn.Sigmoid()
#         self.si_S = nn.Sigmoid()
#         self.so_f = nn.Sigmoid()
#         self.combine = nn.Linear(128 * 4, 128)
#         self.ln = nn.LayerNorm(128)
#         self.drop = nn.Dropout(p=dropout)


#     def forward(self, LM_fea, Sty_fea):

#         Sty_fea_norm = Sty_fea * (abs(torch.mean(LM_fea))/abs(torch.mean(Sty_fea)))
#         f_h = torch.cat((LM_fea.unsqueeze(1), Sty_fea_norm.unsqueeze(1)), dim=1)

#         f_att = torch.mean(f_h, dim=1)
#         f_att = self.so_f(f_att)
#         fus_fea = torch.cat((LM_fea, Sty_fea, LM_fea * f_att, Sty_fea * f_att), dim=1)
#         # print(self.combine(fus_fea))
#         fus_fea = self.combine(fus_fea)

#         return fus_fea
    


class Fusion(nn.Module):
    def __init__(self, hidden_dim=160, dropout=0.05, num_heads=4):
        super(Fusion, self).__init__()

        # 注意力层
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
        # Sigmoid 用于融合
        self.si_L = nn.Sigmoid()
        self.si_S = nn.Sigmoid()
        self.so_f = nn.Sigmoid()

        # 线性变换（拼接后降维）
        self.combine = nn.Linear(hidden_dim * 4, hidden_dim)
        
        # 归一化
        self.ln = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(p=0.05)

    def forward(self, LM_fea, Sty_fea):
        # 归一化 Sty_fea
        Sty_fea_norm = Sty_fea * (abs(torch.mean(LM_fea)) / abs(torch.mean(Sty_fea)))

        # 计算注意力机制
        fea_stack = torch.stack([LM_fea, Sty_fea_norm], dim=1)  # batch_size × 2 × hidden_dim
        attn_output, _ = self.attention(fea_stack, fea_stack, fea_stack)  # Self-Attention
        
        # 计算融合权重
        f_att = torch.mean(attn_output, dim=1)
        f_att = self.so_f(f_att)  # 归一化注意力权重

        # 融合特征
        fus_fea = torch.cat((LM_fea, Sty_fea, LM_fea * f_att, Sty_fea * f_att), dim=1)
        
        # 线性变换 + 归一化
        fus_fea = self.combine(fus_fea)
        fus_fea = self.ln(fus_fea)
        fus_fea = self.drop(fus_fea)

        return fus_fea


# class Fusion(nn.Module):
#     def __init__(self, hidden1, hidden2, dropout=0.05):
#         super(Fusion, self).__init__()
        
#         # 注意力机制的加权系数（128 -> 1）
#         self.attn_weight = nn.Parameter(torch.Tensor(128, 1))  # 对应输入特征维度

#         # 用于特征融合的全连接层
#         self.combine = nn.Linear(128 + 128, 128)  # 输入维度是 128 + 128
#         self.ln = nn.LayerNorm(128)
#         self.drop = nn.Dropout(p=dropout)

#         # 初始化参数
#         nn.init.xavier_uniform_(self.attn_weight)

#     def forward(self, dti1d_feature, dti2d_feature):
#         # 计算加权系数
#         weight_1D = torch.sigmoid(torch.matmul(dti1d_feature, self.attn_weight))  # 对 1D 特征的加权系数
#         weight_2D = torch.sigmoid(torch.matmul(dti2d_feature, self.attn_weight))  # 对 2D 特征的加权系数

#         # 对两种特征应用注意力加权
#         dti1d_feature_weighted = dti1d_feature * weight_1D
#         dti2d_feature_weighted = dti2d_feature * weight_2D

#         # 融合加权后的特征
#         fus_fea = torch.cat((dti1d_feature_weighted, dti2d_feature_weighted), dim=-1)

#         # 使用全连接层进行特征整合
#         fus_fea = self.combine(fus_fea)
#         fus_fea = self.ln(fus_fea)
#         fus_fea = self.drop(fus_fea)

#         return fus_fea





class Discriminator(nn.Module):
    def __init__(self, max_iter, source_number):
        super(Discriminator, self).__init__()
        self.fc1 = Parameter(torch.Tensor(160, 128))#nn.Linear(75, 128)
        self.fc2 = Parameter(torch.Tensor(160, 128))
        self.fc3 = Parameter(torch.Tensor(128, source_number))  # nn.Linear(128, source_number)
        self.drop_lm = nn.Dropout(0.0)
        self.drop_sty = nn.Dropout(0.0)
        self.drop = nn.Dropout(0.5)
        self.grl_layer = GRL(10000)
        self.grl_layer2 = GRL(100)
        self.source_number = source_number
        # self.device = device

    def forward(self, feature1, feature2):
        device = feature1.device
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


import torch
import torch.nn as nn

class Conv1DExpand(nn.Module):
    def __init__(self, input_length=160, target_length=1300, channels=85, kernel_size=5, stride=4, padding=2):
        super(Conv1DExpand, self).__init__()
        
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels=channels, out_channels=channels,  # 确保 in_channels = out_channels
            kernel_size=kernel_size, stride=stride, padding=padding, output_padding=1
        )
        
        # 额外使用 AdaptiveAvgPool1d 确保输出长度精确为 1300
        self.pool = nn.AdaptiveAvgPool1d(target_length)

    def forward(self, drugConv):
        drugConv = drugConv.permute(0, 2, 1)  # (batch, 85, 160)
        expanded_feature = self.conv_transpose(drugConv)  # (batch, 85, ≈1300)
        expanded_feature = self.pool(expanded_feature)  # (batch, 85, 1300) 进行最终修正
        return expanded_feature.permute(0, 2, 1)  # (batch, 1300, 85)




class FusionModel(nn.Module):
    def __init__(self, cnn_dim=85, gcn_dim=75, fused_dim=128, dropout=0.1):
        super(FusionModel, self).__init__()

        # 替换为 1D 卷积扩展
        self.feature_expand = Conv1DExpand(input_length=160, target_length=1300, channels=cnn_dim)

        # 注意力权重
        self.attn_cnn = nn.Parameter(torch.Tensor(cnn_dim, 1))
        self.attn_gcn = nn.Parameter(torch.Tensor(gcn_dim, 1))

        self.fc = nn.Linear(cnn_dim + gcn_dim, fused_dim)
        self.ln = nn.LayerNorm(fused_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.attn_cnn)
        nn.init.xavier_uniform_(self.attn_gcn)

    def forward(self, drugConv, GCN_Feature):
        # 1️⃣ drugConv 升维
        drugConv_expanded = self.feature_expand(drugConv)  # (batch, 1300, 85)

        # import pdb;pdb.set_trace()
        
        # 2️⃣ 计算注意力权重
        weight_cnn = torch.sigmoid(torch.matmul(drugConv_expanded, self.attn_cnn))  # (batch, 1300, 1)
        weight_gcn = torch.sigmoid(torch.matmul(GCN_Feature, self.attn_gcn))  # (batch, 1300, 1)

        # 3️⃣ 加权特征
        drugConv_weighted = drugConv_expanded * weight_cnn
        GCN_Feature_weighted = GCN_Feature * weight_gcn

        # 4️⃣ 融合
        fused_features = torch.cat((drugConv_weighted, GCN_Feature_weighted), dim=-1)  # (batch, 1300, 85+75)

        # 5️⃣ 通过全连接层
        fused_features = self.fc(fused_features)
        fused_features = self.ln(fused_features)
        fused_features = self.dropout(fused_features)

        return fused_features





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
        self.FC_out2 = DT_LeNet(75, 0.0, 2, 3)
        # self.FC_out2 = DT_LeNet(128, 0.0, 2, 3)
        self.dis = Discriminator(100, 3)
        # self.fusion = Fusion(160, 160)
        # self.fusion = Fusion(160, 75)
        self.fusion = Fusion(128)
        self.lamda = Parameter(torch.Tensor([0.8, 0.1, 0.1]))
        self.embed_protein = nn.Embedding(26, 75)
        self.encoder_protein_GNN = Encoder(75, 75, 15, 5, groups=1, pad1=15, pad2=7)
        self.layer_gnn = 3
        layer_gnn = 3
        self.W_dnn = nn.ModuleList([nn.Linear(75, 75)
                                    for _ in range(layer_gnn)])
        self.W_pnn = nn.ModuleList([nn.Linear(75, 75)
                                    for _ in range(layer_gnn)])
        hidden2 = 75
        self.gnn_act = nn.GELU()
        self.bn_A = nn.ModuleList([nn.BatchNorm1d(hidden2) for _ in range(layer_gnn)])
        self.bn_B = nn.ModuleList([nn.BatchNorm1d(hidden2) for _ in range(layer_gnn)])
        self.bn_C = nn.ModuleList([nn.BatchNorm1d(hidden2) for _ in range(layer_gnn)])
        self.bn_D = nn.ModuleList([nn.BatchNorm1d(hidden2) for _ in range(layer_gnn)])
        self.gnn_drop = nn.Dropout(p=0.05)
        self.G_A = nn.ModuleList(
            [nn.Conv1d(in_channels=hidden2, out_channels=hidden2, kernel_size=3, padding=1, groups=hidden2, bias=False)
             for _ in range(layer_gnn)])
        self.soft_1 = nn.Softmax(-1)
        self.soft_2 = nn.Softmax(-1)
        self.soft_3 = nn.Softmax(-1)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.dropout = nn.Dropout(p=0.05)
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.conv*8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)
        
        self.DTI_feature = nn.ModuleList([nn.Linear(128, 128) for _ in range(2)])
        self.act = nn.ReLU()
        self.DTI_Pre = nn.Linear(128, 2)
        self.fusion_2d = FusionModel()

    def Style_Exract(self, df, da, pf, layer):
        for i in range(layer):  # 多层迭代，逐层增强特征交互
            # 1. 基础特征变换（线性层+激活函数）
            ds = self.gnn_act(self.W_dnn[i](df))  # 药物特征线性变换 + GELU激活
            ps = self.gnn_act(self.W_pnn[i](pf))  # 蛋白质特征线性变换 + GELU激活

            # 2. 蛋白质特征的图卷积更新（基于药物特征的风格）
            # G_A：对蛋白质特征做深度可分离卷积，捕捉局部结构
            dg_A = self.bn_A[i](self.G_A[i](ps.permute(0, 2, 1))).permute(0, 2, 1)
            # G_CB：蛋白质-药物的注意力矩阵（计算二者的关联权重）
            G_CB = torch.matmul(pf, df.permute(0, 2, 1))
            # dg_B：基于注意力的药物特征加权更新蛋白质特征
            dg_B = self.bn_B[i](torch.matmul(self.soft_1(G_CB), ds).permute(0, 2, 1)).permute(0, 2, 1)
            # 蛋白质特征更新：原始特征 + 卷积特征 + 注意力加权的药物特征
            pf = dg_A + dg_B + pf

            # 3. 药物特征的图卷积更新（基于蛋白质特征的风格）
            # dg_C：基于药物邻接矩阵的图卷积更新（利用分子结构信息）
            dg_C = self.bn_C[i](torch.matmul(da, ds).permute(0, 2, 1)).permute(0, 2, 1)
            # G_BC：药物-蛋白质的注意力矩阵（反向关联权重）
            G_BC = torch.matmul(df, pf.permute(0, 2, 1))
            # dg_D：基于注意力的蛋白质特征加权更新药物特征
            dg_D = self.bn_D[i](torch.matmul(self.soft_2(G_BC), ps).permute(0, 2, 1)).permute(0, 2, 1)
            # 药物特征更新：原始特征 + 邻接矩阵卷积特征 + 注意力加权的蛋白质特征
            df = dg_C + dg_D + df

        # 4. 最终融合：拼接更新后的蛋白质特征和药物特征
        dt = torch.cat((pf, df), dim=1)
        return dt

    def forward(self, drug, protein,protein_LM,molecule_LM,molecule_atoms_new,molecule_adjs_new):
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
        
        
        """2.DTI 1D feature with pretrain language model"""
        proteins_acids_LM = self.encoder_protein_LM(protein_LM.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1)
        molecule_smiles_LM = self.encoder_drug(molecule_LM.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1)

        DT_1D_Feature = torch.cat((proteins_acids_LM, molecule_smiles_LM), 1)
        DT_1D_P_att = self.dropout(self.Informer_blocks[0](DT_1D_Feature, proteins_acids_LM, proteins_acids_LM, None))
        DT_1D_D_att = self.dropout(self.Informer_blocks[1](DT_1D_Feature, molecule_smiles_LM, molecule_smiles_LM, None))
        DT_1D_Feature = DT_1D_P_att + DT_1D_D_att
        
        
        """3.DTI 2D feature with Graph Nerual Networks"""
        device = protein.device
        N = protein.shape[0]
        # 初始化蛋白质GNN特征张量：[batch_size, protein序列长度, 75]，75是GNN特征维度
        proteins_acids_GNN = torch.zeros((protein.shape[0], protein.shape[1], 75), device=device)
        # 初始化DTI 2D特征最终输出张量：[batch_size, 1300, 75]，1300是固定长度（对齐特征维度）
        DT_2D_Feature = torch.zeros((N, 1300, 75), device=device) # b 1300 d 1400
        #循环嵌入蛋白质序列（逐样本处理）
        for i in range(N):
            proteins_acids_GNN[i, :, :] = self.embed_protein(torch.LongTensor(protein[i].to('cpu').numpy()).cuda())
        #编码器处理 GNN 特征
        proteins_acids_GNN = self.encoder_protein_GNN(proteins_acids_GNN.permute(0, 2, 1)).permute(0, 2, 1)
        
        #双向特征融合
        DT_2D_F = self.Style_Exract(molecule_atoms_new,molecule_adjs_new, proteins_acids_GNN, self.layer_gnn)
        #蛋白质GNN特征张量→DTI 2D特征张量
        t = DT_2D_F.shape[1]
        if t < 1300:
            DT_2D_Feature[:, 0:t, :] = DT_2D_F
        else:
            DT_2D_Feature = DT_2D_F[:, 0:1300, :]





        #4.通过 DT_LeNet（CNN+FC 组合）对 1D/2D 特征解码，输出初步预测和中间特征
        dti1d, dti1d_feature, LM_feature = self.FC_out1(DT_1D_Feature)
        dti2d, dti2d_feature, GRL_feature = self.FC_out2(DT_2D_Feature)

        #5.通过多头注意力 + 线性变换融合 1D/2D 特征：
        DTI = self.fusion(dti1d_feature, dti2d_feature)

        #6.对融合特征再次通过两层 FC 增强，输出最终预测：
        for i in range(2):
            DTI = self.act(self.DTI_feature[i](DTI))
        predict = self.DTI_Pre(DTI)
        

        return predict,dti1d, dti2d, dti1d_feature, dti2d_feature


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