# cs4100-pcr-gradcam-prediction

** Contributors: Christian Garcia, Tommaso Maga, Yu-Chun Ou, Peter SantaLucia

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

Works Cited
Toss, Angela, et al. “Predictive factors for relapse in triple-negative breast cancer patients without pathological complete response after neoadjuvant chemotherapy.” Frontiers in Oncology, vol. 12, 1 Dec. 2022, https://doi.org/10.3389/fonc.2022.1016295. 

