# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

MCANet (Multi-head Cross-Attention Network) predicts **Drug–Target Interactions (DTI)** — a binary classification over (drug SMILES, protein sequence) pairs. This snapshot contains the **LLM + Graph** enhanced variant, driven by `RunModel_LLM.py` and `model_graph/`.

## Entry Point

- `main.py` — CLI entry. Runs the LLM variant via `RunModel_LLM.run_model`.
- `RunModel_LLM.py` — training pipeline (K-fold CV, early stopping, CyclicLR, PolyLoss). Imports `model_graph.model_lmm_dis.MCANet`.
- `config.py` — hyperparameters (`hyperparameter` class).
- `LossFunction.py` — `PolyLoss` and `CELoss`.

## Architecture (`model_graph/model_lmm_dis.py`)

Dual pathway + final fusion:

1. **Semantic (1D) pathway** — ProtBERT (`Rostlab/prot_bert_bfd`, 1024-d) + SMILES-BERT (`seyonec/PubChem10M_SMILES_BPE_450k`, 768-d) → dense-CNN `Encoder` → Informer `ProbAttention` sparse cross-attention → `DT_LeNet`.
2. **Structural (2D / graph) pathway** — SMILES → molecular graph (RDKit atom features + adjacency) → GNN; protein sequence → embedding → `Encoder`; `Style_Exract` performs bidirectional drug↔protein graph fusion → `DT_LeNet`.
3. **Final fusion** — multi-head attention `Fusion` + two FC layers → binary logits.

## Data Flow

1. Raw `DataSets/*.txt` lines: `drug_id protein_id SMILES protein_sequence label(0/1)`.
2. LM embeddings are precomputed with `utils/prepare.py` (`DTA_datasets`) and cached as `*.pkl`.
3. `utils/DataSets_graph.py` builds batches (char-encoded sequences, molecular graph, cached LM features) and loads `Davis_data_p_LM.pkl` / `Davis_data_d_LM.pkl` at import time.
4. `utils/DataPrepare.py` handles K-fold splitting; `utils/TestModel.py` and `utils/EarlyStoping.py` evaluate.

## Datasets

Davis, KIBA, DrugBank, Enzyme, GPCRs, ion_channel (raw files are not committed).

## Common Commands

```bash
python utils/prepare.py   # generate LM embedding .pkl caches
python main.py Davis      # train + test on Davis (5-fold CV)
```

## Important Notes / Caveats

- `RunModel_LLM.py` hard-codes `DEVICE = torch.device("cuda")`.
- `utils/DataSets_graph.py` hard-codes absolute `.pkl` paths at import; regenerate with `prepare.py` and adjust for your machine.
- `PolyLoss.forward` assumes a batch size of 16.
- `main.py` lists `davis` (lowercase) while `RunModel_LLM` asserts `Davis` (capitalized) — use the capitalized spelling.
- The original CNN + cross-attention `MCANet` (`model.py` / `RunModel.py`) is not present in this snapshot; only the LLM/graph variant is included.
