import torch
from torch.utils.data import Dataset
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import scipy.sparse as sp
from .graph_features import atom_features
from collections import defaultdict
from rdkit.Chem.rdchem import BondType
from transformers import AutoModel, AutoTokenizer
import pickle

# import torch.multiprocessing as mp
# import torch

# # 设定 multiprocessing 启动方式为 spawn
# mp.set_start_method('spawn', force=True)

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

CHARPROTLEN = 25

BONDTYPE_TO_INT = defaultdict(
    lambda: 0,
    {
        BondType.SINGLE: 0,
        BondType.DOUBLE: 1,
        BondType.TRIPLE: 2,
        BondType.AROMATIC: 3
    }
)


def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN, dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

def smile_to_graph(smile):
    # 步骤1：将SMILES字符串解析为RDKit的Mol对象（分子结构）
    molecule = Chem.MolFromSmiles(smile)
    # 步骤2：获取分子中原子总数 + 所有原子对象
    n_atoms = molecule.GetNumAtoms()
    atoms = [molecule.GetAtomWithIdx(i) for i in range(n_atoms)]
    # 步骤3：生成邻接矩阵（图的拓扑结构）
    adjacency = Chem.rdmolops.GetAdjacencyMatrix(molecule)
    # 步骤4：生成节点特征矩阵（每个原子的属性）
    node_features = np.array([atom_features(atom) for atom in atoms])
    # 步骤5：生成边特征矩阵（每个化学键的属性）
    n_edge_features = 4
    edge_features = np.zeros([n_atoms, n_atoms, n_edge_features])
    for bond in molecule.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = BONDTYPE_TO_INT[bond.GetBondType()]
        edge_features[i, j, bond_type] = 1
        edge_features[j, i, bond_type] = 1
    #返回邻接矩阵（图的拓扑结构）、节点特征矩阵（每个原子的属性）
    return node_features, adjacency

def process_molecule_atoms(molecule_atoms):
    
    atom_num = []
    atoms_len = 0
    N = len(molecule_atoms)
    
    # 计算最大原子数量
    for atom in molecule_atoms:
        atom_num.append(atom.shape[0])
        if atom.shape[0] > atoms_len:
            atoms_len = atom.shape[0]
            
    molecule_atoms_new = torch.zeros(N, atoms_len, 75)

    for i, atom in enumerate(molecule_atoms):
        a_len = atom.shape[0]
        molecule_atoms_new[i, :a_len, :] = atom

    return molecule_atoms_new


class CustomDataSet(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, item):
        return self.pairs[item]

    def __len__(self):
        return len(self.pairs)


# file_path_p = "/home/4paradigm/MCANet/Davis_data/KIBA/p_LM.pkl"
# file_path_d = "/home/4paradigm/MCANet/Davis_data/KIBA/d_LM.pkl"

file_path_p = "D:\python_study\MCA\Davis_data_p_LM.pkl"
file_path_d = "D:\python_study\MCA\Davis_data_d_LM.pkl"


# 使用 pickle 读取数据
with open(file_path_p, 'rb') as file:
    p_LM = pickle.load(file)
    
with open(file_path_d, 'rb') as file:
    d_LM = pickle.load(file)


def collate_fn(batch_data):
    N = len(batch_data)
    drug_ids, protein_ids = [], []
    p_l = 1200
    d_l = 100
    sequences=[]
    smiles=[]
    d_LLMs=[]
    p_LLMs=[]
    # supports = []
    compound_max = 100
    protein_max = 1200
    atoms_len = 0
    compound_new = torch.zeros((N, compound_max), dtype=torch.long)
    protein_new = torch.zeros((N, protein_max), dtype=torch.long)
    labels_new = torch.zeros(N, dtype=torch.long)
    atom_features = torch.zeros((N, compound_max, 75), dtype=torch.long)  # Change 39 to compound_max
    adjs = torch.zeros((N, compound_max, compound_max), dtype=torch.long)  # Change 36 to compound_max
    
    
    for i, pair in enumerate(batch_data):
        pair = pair.strip().split()
        drug_id, protein_id, compoundstr, proteinstr, label = pair[-5], pair[-4], pair[-3], pair[-2], pair[-1]
        # drug_ids.append(drug_id)
        # protein_ids.append(protein_id)
        sequences.append(proteinstr)
        smiles.append(compoundstr)
        atom_feature, adj = smile_to_graph(compoundstr)
        
        
        d_LLM=d_LM[compoundstr]
        p_LLM = p_LM[proteinstr]
        d_LLMs.append(d_LLM)
        p_LLMs.append(p_LLM)
        
        # 将SMILES字符串转化为向量
        compoundint = torch.from_numpy(label_smiles(compoundstr, CHARISOSMISET, compound_max))
        compound_new[i] = compoundint        
        
        # 获取蛋白质的序列向量
        proteinint = torch.from_numpy(label_sequence(proteinstr, CHARPROTSET, protein_max))
        protein_new[i] = proteinint
        
        # 处理标签
        label = float(label)
        labels_new[i] = int(label)
    
    atom_num = []
    for compoundstr in smiles:
        atom_feature, adj = smile_to_graph(compoundstr)
        atom_num.append(atom_feature.shape[0])
        if atom_feature.shape[0] >= atoms_len:
            atoms_len = atom_feature.shape[0]

    molecule_atoms_new = torch.zeros(N, atoms_len, 75)
    i = 0
    for compoundstr in smiles:
        atom_feature, adj = smile_to_graph(compoundstr)
        atom_feature = torch.tensor(atom_feature)
    # for atom in molecule_atoms:
        a_len = atom_feature.shape[0]
        molecule_atoms_new[i, :a_len, :] = atom_feature
        i += 1

    molecule_adjs_new = torch.zeros(N, atoms_len, atoms_len)
    i = 0
    for compoundstr in smiles:
        atom_feature, adj = smile_to_graph(compoundstr)
        adj = torch.tensor(adj)
        adj = torch.tensor(adj)
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len)
        molecule_adjs_new[i, :a_len, :a_len] =adj
        i += 1
    
    
    atom_num = []
        
    protein_LM = torch.zeros(N, p_l, 1024)
    molecule_LM = torch.zeros(N, d_l, 768)

    for i in range(N):
        C_L = d_LLMs[i].shape[0]
        if C_L >= 100:
            molecule_LM[i, :, :] = torch.tensor(d_LLMs[i][0:100, :])
        else:
            molecule_LM[i, :C_L, :] = torch.tensor(d_LLMs[i])
        P_L = p_LLMs[i].shape[0]
        if P_L >= 1200:
            protein_LM[i, :, :] = torch.tensor(p_LLMs[i][0:1200, :])
        else:
            protein_LM[i, :P_L, :] = torch.tensor(p_LLMs[i])

    # 返回处理后的分子、支持矩阵和标签
    return sequences,smiles,compound_new, protein_new, labels_new,protein_LM,molecule_LM,molecule_atoms_new,molecule_adjs_new
