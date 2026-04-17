# cs4100-pcr-gradcam-prediction

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.png)](https://creativecommons.org/licenses/by-nc/4.0/)

**Contributors:** Christian Garcia, Tommaso Maga, Yu-Chun Ou, Peter SantaLucia

## License

This project is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) to comply with the licensing terms of the BreastDCEDL dataset, which itself adopts CC BY-NC 4.0 as the most restrictive license among its three source datasets from TCIA. Specifically, while I-SPY 1 (CC BY 3.0) and I-SPY 2 (CC BY 4.0) permit commercial use, the Duke Breast Cancer MRI dataset (CC BY-NC 4.0) does not. As a derivative work integrating all three sources, this project inherits that restriction. Academic use, research, and adaptation with attribution are welcome.

## Purpose

The group has decided to predict patient breast cancer outcomes using MRI scans and patient metadata. An accurate measure of these outcomes is given by the pathological complete response (PCR) metric for triple-negative breast cancer (TNBC). That is, the complete remission of an invasive cancer that is found in a tissue sample. For TNBC Patients with PCR, 90% of patients experience event-free survival (EFS) over a period of three years, while only 67% experienced EFS over the same period (Toss, et al.). By predicting PCR for some patients, we are able to accurately predict patient outcomes over a period of three years.
By having an accurate prediction of a patient’s PCR, doctors can preemptively consider other treatment plans/evaluate the need for surgery for some patient. Given the aggressiveness of TNBC, it’s widely studied, leading to many high-quality multiparametric MRI datasets available for academic use. With these datasets, we are able to train some model on 3D volumetric data that will help us predict long-term patient outcomes.

## Existing Approaches

- [Nature Scientific Reports (2025)](https://www.nature.com/articles/s41598-025-97763-0): Trained machine learning models to predict survival rates for patients undergoing breast cancer treatments. _(Note: Our project specifically builds on this domain by focusing on deep learning for 3D volumetric data and adding interpretability)._

## Datasets & Methodology

- **Dataset:** [BreastDCEDL](https://github.com/naomifridman/BreastDCEDL) A deep-learning-ready, labeled dataset that combines dynamic contrast-enhanced (DCE) MRI scans from three major clinical trials.
- **Environment:** Google Colab.
- **Architecture:** 3D Convolutional Neural Network (CNN) trained on volumetric MRI scans.
- **Interpretability:** We will implement **HiResCAM** to visualize the model's decision-making process. Because medical decisions require high transparency, HiResCAM will ensure the generated heatmaps are mathematically faithful to the model's internal weights.
- **Stretch Goal:** If time allows, we will implement custom positive/negative filters on the HiResCAM outputs. This toggle will allow clinicians to isolate the specific tissue features acting as positive contributors (evidence _for_ pCR) versus negative contributors (evidence _against_ pCR).

## Codebase Organization

```
cs4100-pcr-gradcam-prediction/
├── dataset.py              # PyTorch Dataset class — loads & preprocesses DCE-MRI NIfTI volumes
├── train.py                # Training pipeline (PyTorch Lightning) — defines PcrCNN, runs train/val/test
├── visualize.py            # HiResCAM visualization — overlays attention maps on MRI slices
├── requirements.txt        # Python dependencies
├── pay_attn/
│   ├── hirescam.py         # HiResCAM algorithm implementation (gradient × activation)
│   ├── model_outputs.py    # Hook-based activation/gradient extraction from PcrCNN
│   └── test.py             # Standalone test for HiResCAM on a single sample
├── model_samples/
│   ├── model_best_auroc.pth    # Pre-trained checkpoint (best validation AUROC)
│   └── model_best_loss.pth     # Pre-trained checkpoint (best validation loss)
├── data/                   # Dataset directory (not tracked in git — see Environment Setup below)
│   ├── BreastDCEDL_metadata_min_crop.csv
│   ├── BreastDCEDL_ISPY1_min_crop/dce/
│   ├── BreastDCEDL_ISPY2_min_crop/dce/
│   └── BreastDCEDL_DUKE_min_crop/crop_min_dce/
└── source/                 # Sphinx documentation source
```

**Data flow:** `dataset.py` loads NIfTI volumes from `data/`, stacks 3 DCE time-point volumes into a `(3, 32, 256, 256)` tensor, and normalizes to `[0, 1]`. `train.py` uses this dataset to train `PcrCNN`, a 4-block 3D CNN that outputs a binary pCR logit. `visualize.py` and `pay_attn/` apply HiResCAM to the trained model to produce interpretable attention overlays.

---

## Environment Setup

**Requirements:** Python 3.12

1. Clone the repository:

   ```bash
   git clone https://github.com/Hypurl/cs4100-pcr-gradcam-prediction.git
   cd cs4100-pcr-gradcam-prediction
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** `triton-windows` in `requirements.txt` is Windows-only. Remove it before installing on macOS or Linux.

4. Obtain the dataset: Request access to [BreastDCEDL](https://github.com/naomifridman/BreastDCEDL) and place the data folders under `data/` matching the directory structure shown above. The metadata CSV (`BreastDCEDL_metadata_min_crop.csv`) must also be placed in `data/`.

---

## Running the Code

### Training

```bash
python train.py
```

Trains the 3D CNN on the BreastDCEDL dataset. Best checkpoints are saved to `checkpoints/` and exported as `model_best_auroc.pth` and `model_best_loss.pth`.

### Visualization (HiResCAM)

```bash
python visualize.py
```

Loads `model_samples/model_best_auroc.pth` and generates HiResCAM attention map overlays on MRI slices for a sample patient, saving output PNGs to the project root.

### Testing the HiResCAM Module

```bash
python pay_attn/test.py
```

Runs a quick sanity check of the HiResCAM pipeline on test split sample 0, printing heatmap shape and statistics.

### Dataset Inspection

```bash
python dataset.py
```

Runs the built-in test block: loads the first sample and displays a 9-slice grid visualization.

---

## Works Cited

Draelos, Rachel Lea, and Lawrence Carin. “Use HiResCAM Instead of Grad-CAM for Faithful Explanations of Convolutional Neural Networks.” _arXiv_, 17 Nov. 2020, arxiv.org/abs/2011.08891.

Fridman, Naomi, et al. “BreastDCEDL: A Deep Learning–Ready Breast DCE-MRI Dataset.” _Zenodo_, 9 June 2025, doi.org/10.5281/zenodo.15627233.

Toss, Angela, et al. “Predictive Factors for Relapse in Triple-Negative Breast Cancer Patients without Pathological Complete Response after Neoadjuvant Chemotherapy.” _Frontiers in Oncology_, vol. 12, 1 Dec. 2022, doi.org/10.3389/fonc.2022.1016295.
