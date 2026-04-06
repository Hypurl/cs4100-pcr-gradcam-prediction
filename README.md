# cs4100-pcr-gradcam-prediction

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.png)](https://creativecommons.org/licenses/by-nc/4.0/)

**Contributors:** Christian Garcia, Tommaso Maga, Yu-Chun Ou, Peter SantaLucia

## License
This project is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) to comply with the licensing terms of the BreastDCEDL dataset, which itself adopts CC BY-NC 4.0 as the most restrictive license among its three source datasets from TCIA. Specifically, while I-SPY 1 (CC BY 3.0) and I-SPY 2 (CC BY 4.0) permit commercial use, the Duke Breast Cancer MRI dataset (CC BY-NC 4.0) does not. As a derivative work integrating all three sources, this project inherits that restriction. Academic use, research, and adaptation with attribution are welcome.

## Purpose
The group has decided to predict patient breast cancer outcomes using MRI scans and patient metadata. An accurate measure of these outcomes is given by the pathological complete response (PCR) metric for triple-negative breast cancer (TNBC). That is, the complete remission of an invasive cancer that is found in a tissue sample. For TNBC Patients with PCR, 90% of patients experience event-free survival (EFS) over a period of three years, while only 67% experienced EFS over the same period (Toss, et al.). By predicting PCR for some patients, we are able to accurately predict patient outcomes over a period of three years. 
By having an accurate prediction of a patient’s PCR, doctors can preemptively consider other treatment plans/evaluate the need for surgery for some patient. Given the aggressiveness of TNBC, it’s widely studied, leading to many high-quality multiparametric MRI datasets available for academic use. With these datasets, we are able to train some model on 3D volumetric data that will help us predict long-term patient outcomes. 

## Existing Approaches
* [Nature Scientific Reports (2025)](https://www.nature.com/articles/s41598-025-97763-0): Trained machine learning models to predict survival rates for patients undergoing breast cancer treatments. *(Note: Our project specifically builds on this domain by focusing on deep learning for 3D volumetric data and adding interpretability).*

## Datasets & Methodology
* **Dataset:** [BreastDCEDL](https://github.com/naomifridman/BreastDCEDL) A deep-learning-ready, labeled dataset that combines dynamic contrast-enhanced (DCE) MRI scans from three major clinical trials.
* **Environment:** Google Colab.
* **Architecture:** 3D Convolutional Neural Network (CNN) trained on volumetric MRI scans.
* **Interpretability:** We will implement **HiResCAM** to visualize the model's decision-making process. Because medical decisions require high transparency, HiResCAM will ensure the generated heatmaps are mathematically faithful to the model's internal weights. 
* **Stretch Goal:** If time allows, we will implement custom positive/negative filters on the HiResCAM outputs. This toggle will allow clinicians to isolate the specific tissue features acting as positive contributors (evidence *for* pCR) versus negative contributors (evidence *against* pCR).

## Works Cited
Draelos, Rachel Lea, and Lawrence Carin. “Use HiResCAM Instead of Grad-CAM for Faithful Explanations of Convolutional Neural Networks.” *arXiv*, 17 Nov. 2020, arxiv.org/abs/2011.08891.

Fridman, Naomi, et al. “BreastDCEDL: A Deep Learning–Ready Breast DCE-MRI Dataset.” *Zenodo*, 9 June 2025, doi.org/10.5281/zenodo.15627233.

Toss, Angela, et al. “Predictive Factors for Relapse in Triple-Negative Breast Cancer Patients without Pathological Complete Response after Neoadjuvant Chemotherapy.” *Frontiers in Oncology*, vol. 12, 1 Dec. 2022, doi.org/10.3389/fonc.2022.1016295.

