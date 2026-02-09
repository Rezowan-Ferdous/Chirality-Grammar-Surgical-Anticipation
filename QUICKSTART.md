# PTG Framework - Quick Start Guide

## 🚀 Complete Workflow

### 1. Preprocess Cholec80 Dataset
```bash
python scripts/preprocess_cholec80.py \
    --data_root D:/Datasets/cholec80 \
    --output_dir data/cholec80_annotations
```

### 2. Extract V-JEPA Features
```bash
python scripts/extract_vjepa_features.py \
    --video_dir D:/Datasets/cholec80/videos \
    --output_dir data/features/cholec80 \
    --num_frames 16
```

### 3. Induce Grammar
```bash
python scripts/induce_grammar.py \
    --corpus_path data/cholec80_annotations \
    --output_path grammars/cholec80_ptg.pkl \
    --min_ngram_freq 3 \
    --markov_order 2
```

### 4. Train Phase 1 (Supervised)
```bash
python scripts/train_ptg.py \
    --config configs/ptg_chirality.yaml \
    --phase 1 \
    --output_dir outputs/phase1
```

### 5. Train Phase 2 (Grammar-Regularized)
```bash
python scripts/train_ptg.py \
    --config configs/ptg_chirality.yaml \
    --phase 2 \
    --grammar_path grammars/cholec80_ptg.pkl \
    --output_dir outputs/phase2
```

### 6. Evaluate
```bash
python scripts/evaluate.py \
    --model_path outputs/phase2/ptg_phase2_final.pth \
    --config configs/ptg_chirality.yaml \
    --test_split test \
    --output_file results/evaluation.json
```

### 7. Inference
```bash
python scripts/inference.py \
    --model_path outputs/phase2/ptg_phase2_final.pth \
    --video_path videos/test_video.mp4 \
    --config configs/ptg_chirality.yaml \
    --grammar_path grammars/cholec80_ptg.pkl \
    --max_steps 20
```

---

## 📊 Key Metrics

- **MoC Accuracy**: Mean over Classes at horizons {1, 5, 10, 20, 50}
- **Chirality F1**: Precision/Recall for chiral actions
- **Duration MAE**: Mean absolute error for duration prediction
- **Edit Distance**: Sequence-level similarity

---

## 🔧 Module Overview

| Module | Purpose | Key Files |
|--------|---------|-----------|
| `src/grammar` | Grammar induction | `ngram_miner.py`, `transition_model.py`, `chirality_lexicon.py` |
| `src/models` | Neural networks | `vjepa_extractor.py`, `futr_decoder.py`, `ptg_model.py` |
| `src/training` | Loss functions | `losses.py` |
| `src/data` | Dataset | `cisa_dataset.py` |
| `src/inference` | Parsing | `earley_parser.py` |
| `src/evaluation` | Metrics | `metrics.py` |
| `scripts` | Executables | All `.py` files |

---

## ✅ Status: **COMPLETE**

All 8 phases implemented with 20+ modules and ~4,500 lines of code!
