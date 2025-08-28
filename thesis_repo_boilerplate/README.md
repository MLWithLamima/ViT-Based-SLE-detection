# BMR Detection with Vision Transformers (Thesis)

This repository contains code and assets for the thesis **"A Vision Transformer Based Assistive System for Dermatological Diagnosis in SLE"**.
It compares CNN baselines (e.g., ResNet50, DenseNet121, MobileNetV2) with Vision Transformers (e.g., ViT-Small/DeiT) and explores augmentation and CutMix.

## 📁 Suggested Layout
```
repo-root/
├─ CNNs_Before_Augmentation/
├─ CNNs_After_Augmentation/
├─ CNN_with_CutMix/
├─ ViTs_Before_Augmentation/
├─ ViTs_After_Augmentation/
├─ ViTs_with_cutmix/
├─ models/                 # place .pth here (tracked by Git LFS) 
├─ data/                   # not tracked by git; see below
├─ notebooks/              # Jupyter notebooks (optional)
├─ src/                    # reusable python code (optional)
├─ requirements.txt
└─ README.md
```

> ⚠️ **Datasets are NOT committed.** Keep raw data outside git or manage with DVC.  
> ⚠️ **Models (.pth)** are tracked with Git LFS or uploaded as Release assets.

## 🛠️ Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## ▶️ Reproduce (example)
```bash
# train a baseline (example only)
python src/train_cnn.py --arch resnet50 --epochs 50 --data ./data

# evaluate a saved model
python src/eval.py --weights ./models/Suggested_Best_Model.pth --split test
```

## 🔄 Augmentation & CutMix
- Standard augmentations (flip, rotate, color jitter) via Albumentations
- Optional **CutMix** experiments for ViTs/CNNs

## 📊 Results
Add confusion matrices, ROC curves, and tables here (or link to `results/`).

## 📦 Models
- `models/Suggested_Best_Model.pth` (tracked with LFS)  
  Alternatively: upload to **Releases** and link here.

## 📝 Citation
See `CITATION.cff`.

## 🛡️ License
MIT (see `LICENSE`).