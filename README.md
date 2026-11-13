# MCANet

**Multi-head Cross-Attention Network for Drug–Target Interaction (DTI) prediction.**

MCANet predicts whether a drug (SMILES) interacts with a protein (amino-acid sequence) — a binary classification task. This repository implements an **LLM + Graph enhanced** variant that fuses a *semantic* pathway (pre-trained protein/drug language models) with a *structural* pathway (molecular-graph neural networks), and combines them with multi-head cross-attention.

## Architecture

The model (`model_graph/model_lmm_dis.py`) is a dual-pathway design with three-level fusion.

### 1. Semantic pathway (1D)

1. Encode protein sequences with **ProtBERT** (`Rostlab/prot_bert_bfd`, 1024-d) and drug SMILES with **SMILES-BERT** (`seyonec/PubChem10M_SMILES_BPE_450k`, 768-d).
2. A dense-CNN `Encoder` projects each to 160-d sequence features.
3. **Informer `ProbAttention`** performs *sparse* cross-attention between drug and protein features, keeping only the most informative queries to reduce noise and cost.
4. `DT_LeNet` (residual 1D-CNN + FC) decodes the fused semantic feature into a preliminary prediction.

### 2. Structural pathway (2D / graph)

1. Drug SMILES → molecular graph (RDKit atom features + adjacency matrix); a GNN extracts topology.
2. Protein sequence → embedding → `Encoder` extracts local structural context.
3. `Style_Exract` performs **bidirectional graph fusion**: protein features are updated by drug-attention + convolution, and drug features are updated by adjacency convolution + protein-attention.

### 3. Final fusion

The two pathway features are fused with a **multi-head attention** module (`Fusion`) and passed through two fully-connected layers to produce the final binary logits.

## Datasets

Supported benchmark datasets (raw `.txt` files are **not** included; place them under `DataSets/`):

| Dataset | Task |
| --- | --- |
| Davis | Kinase drug–target binding |
| KIBA | Kinase inhibitor bioactivity |
| DrugBank | Approved drug–target interactions |
| Enzyme | Enzyme–drug interactions |
| GPCRs | GPCR–drug interactions |
| ion_channel | Ion-channel–drug interactions |

Each line has the format:

```
drug_id protein_id SMILES protein_sequence label(0/1)
```

## Requirements

- Python 3.8+
- PyTorch + [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- HuggingFace `transformers`
- RDKit
- numpy, scipy, scikit-learn, tqdm, prefetch_generator

Install with:

```bash
pip install -r requirements.txt
```

## Data preparation

Pre-trained language-model embeddings are precomputed and cached to `.pkl` files before training:

```bash
python utils/prepare.py
```

`prepare.py` runs ProtBERT over protein sequences and SMILES-BERT over drug SMILES, and writes `*_p_LM.pkl` / `*_d_LM.pkl` caches. `utils/DataSets_graph.py` loads these caches at import time (see [notes](#notes)).

## Usage

```bash
# Train + evaluate on a dataset (5-fold CV by default)
python main.py Davis

# CLI options
python main.py <Dataset> -m MCANet -s 114514 -f 5
```

- `<Dataset>`: one of `DrugBank`, `KIBA`, `Davis`, `Enzyme`, `GPCRs`, `ion_channel`
- `-m/--model`: `MCANet`
- `-s/--seed`: random seed (default `114514`)
- `-f/--fold`: number of K-folds (default `5`)

Hyperparameters (learning rate, batch size, kernel sizes, embedding dims, etc.) are centralized in `config.py` (`hyperparameter` class).

## Project structure

```
MCA/
├── main.py                  # CLI entry point
├── RunModel_LLM.py          # training pipeline (K-fold CV, early stopping, CyclicLR)
├── config.py                # hyperparameters
├── LossFunction.py          # PolyLoss and CELoss
├── utils/
│   ├── DataPrepare.py       # K-fold split / shuffle
│   ├── DataSets_graph.py    # dataset + collate_fn (LM + graph features)
│   ├── prepare.py           # LM embedding cache generation
│   ├── graph_features.py    # atom feature extraction
│   ├── TestModel.py         # evaluation
│   ├── EarlyStoping.py      # early stopping
│   └── ShowResult.py        # result reporting
└── model_graph/
    ├── model_lmm_dis.py     # MCANet (LLM + graph)
    ├── model_graph.py       # GNN components
    ├── Informer_block.py    # ProbAttention sparse cross-attention
    ├── GCN.py               # graph convolution
    └── mask.py              # attention masks
```

## Notes

- `RunModel_LLM.py` hard-codes `DEVICE = torch.device("cuda")`; change to `cpu` if no GPU is available.
- `utils/DataSets_graph.py` hard-codes absolute `.pkl` paths at import time (`D:\python_study\MCA\Davis_data_p_LM.pkl` / `Davis_data_d_LM.pkl`). Regenerate with `prepare.py` and adjust the paths for your machine.
- `PolyLoss.forward` assumes a batch size of 16 (see `LossFunction.py`).
- `main.py` lists `davis` (lowercase) in its choices, while `RunModel_LLM` asserts `Davis` (capitalized). Use the capitalized spelling expected by the assertion.

## License

No license is currently specified; all rights are reserved by the author.
