# Enhancing Privacy and Accuracy in Facial Recognition with Synthetic Data Generated by Pre-Trained GANs

#### Author: Raj Shah

#### Institution: The George Washington University, School of Engineering and Applied Science

#### Degree: Doctor of Engineering in Cybersecurity Analytics

#### Paper: TBD

#### Synthetic Data with 1 million images that have been demographically categorized: https://drive.google.com/drive/folders/1SlRpmcte_ka0JtA4vyvCB1OZT4V_DpnK?usp=drive_link

## Overview

Facial recognition systems have become increasingly widespread, leading to significant privacy and demographic bias concerns. This research explores synthetic data as a viable alternative for facial recognition model training, aiming to enhance privacy and reduce bias through the use of Generative Adversarial Networks (GANs).
Objectives

   Generate synthetic facial data using StyleGAN2-ADA with CelebA-HQ dataset.

   Assign demographic attributes (age, gender, race) using the FairFace model.

   Evaluate synthetic data effectiveness by training ArcFace models and comparing with real and hybrid datasets.

   Analyze privacy implications and demographic fairness.

## Methodology

   Data Generation: Synthetic images generated using the pre-trained StyleGAN2-ADA model integrated with InterFaceGAN.

   Dataset Cleaning: Manual and automated checks to ensure image clarity and quality.

   Attribute Assignment: Demographic labels assigned using FairFace to create a balanced representation.

   Model Training: ArcFace models trained separately on synthetic, real (CelebA), and hybrid datasets.

   Performance Evaluation: Metrics including Accuracy, F1 Scores, False Positive Rates (FPR), and Fairness Discrepancy Rate (FDR).

## Key Findings

   Synthetic data achieved comparable performance to real and hybrid datasets in accuracy and F1 scores.

   Demonstrated enhanced privacy protection, eliminating reliance on personally identifiable biometric data.

   Highlighted current limitations in demographic fairness, with synthetic-only datasets showing higher fairness discrepancies.

## Usage Instructions
```python
git clone https://github.com/rshah832/Enhancing-Privacy-and-Accuracy-in-Facial-Recognition-with-Synthetic-Data-Praxis.git
cd your-repository
```


Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the following:

os,
pandas,
numpy,
cv2,
matplotlib,
sklearn,
tensorflow,
insightface,
torch,
deepface
```python
pip install cv2
```

## Pre-requisites

Ensure you have [FairFace](https://github.com/joojs/fairface) within your system.

Ensure you have [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset within your system to mimic the findings.

Ensure you have [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch).


## Contributions

Feel free to make any necessary modifications within the code, there will be numerous that may be required based on your needs. One of the core modifications will be to modify the file path whether it be the input or output paths.

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
