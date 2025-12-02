 Adversarial Attacks on AI Models in Cybersecurity

Author: Huzaifa Anis  
Institution: Illinois Institute of Technology  
Course: ITMS 478 – Research Project Implementation  

---
Overview

This project investigates the **robustness of machine learning models for intrusion detection systems (IDS)** against **adversarial attacks**.  
The implementation builds upon the baseline paper:

The goal is to:
- Recreate the baseline IDS experiment using the **NSL-KDD dataset**
- Apply **adversarial perturbations** (FGSM & PGD)
- Evaluate model performance degradation and visualize it through multiple analytical plots
- Summarize robustness results quantitatively and visually for research reporting

---
Research Objectives

1. **Reproduce baseline model** performance on NSL-KDD dataset.  
2. **Implement adversarial perturbations** (Fast Gradient Sign Method and Projected Gradient Descent).  
3. **Quantify model degradation** in terms of accuracy, precision, recall, and F1-score.  
4. **Visualize degradation and recovery potential** through metrics, heatmaps, and ROC/PR curves.  
5. **Generate a reproducible analysis pipeline** for future research comparison.

---
 Methodology

1. Dataset Preparation  
- Uses **NSL-KDD**, a standard dataset for network intrusion detection.  
- Data preprocessing: label encoding, feature scaling, and train-test split.  

2. Baseline Model Training  
- Model: Random Forest (baseline classifier).  
- Saved model artifact: `models/baseline_model.pkl`  

3. Adversarial Attack Generation  
- **FGSM Attack:** Adds signed perturbations to simulate minimal adversarial noise.  
- **PGD Attack:** Iteratively perturbs test data to simulate stronger, repeated attacks.  

4. Graphical Analysis and Evaluation  
- Evaluates and visualizes model resilience using:
  - Confusion matrices
  - Accuracy/F1 bar plots
  - Heatmaps
  - ROC & PR curves
  - Risk likelihood-impact map  

5. Automated Pipeline  
To simplify replication, a **single command** runs the entire workflow:
python run_pipeline.py

### Loom Recording 

 https://www.loom.com/share/79c9721caf4e4f5096ecd2d5f18a6b70?sid=8186b938-0e8b-4446-93f9-efbe86daa9a6

