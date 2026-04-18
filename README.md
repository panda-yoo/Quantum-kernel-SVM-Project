# Quantum Kernel SVM Project

This project explores quantum kernel based Support Vector Machines (SVMs) and compares multiple feature-map circuits on synthetic and real datasets.

The repository contains two main notebooks:

- QMLproject.ipynb: a focused study on the blood transfusion dataset with quantum and classical SVM comparisons.
- reproducingArticle.ipynb: a broader reproduction of experiments inspired by the article "Comparative performance analysis of quantum feature maps for quantum kernel-based machine learning".

## Notebooks Overview

### 1) QMLproject.ipynb

Goal:
- Build intuition for quantum feature maps on a binary classification task.
- Compare quantum precomputed-kernel SVM decision boundaries against classical SVM kernels.

Main flow:
- Load and inspect the Blood Transfusion dataset.
- Select two weakly correlated features:
	- Recency (months)
	- Time (months)
- Split data into train/test and normalize features to [0, pi].
- Define 4 custom phase functions phi(x, y) and generate feature-map circuits.
- Train quantum kernel matrices with parameter lambda.
- Fit SVM(kernel="precomputed") and evaluate test accuracy.
- Plot decision boundaries for quantum and classical baselines (RBF, linear, polynomial).

Key outputs:
- result/quantum_circuits.jpg
- result/desisionBoundaries.jpg
- result/Desision_Boundaries_classical.jpg

### 2) reproducingArticle.ipynb

Goal:
- Reproduce and extend comparative experiments from the article across multiple datasets and feature maps.

Main flow:
- Generate synthetic datasets:
	- Circles
	- Moons
	- XOR
- Load Breast Cancer Wisconsin (Diagnostic) and select two features:
	- worst texture
	- mean concavity
- Scale features (for most experiments, to [-1, 1]).
- Define six feature-map functions f1..f6.
- Build circuit families that vary by dataset type (circle, moon, XOR, WBC).
- Sweep alpha values in {0.5, 1.0, 2.0, 3.0}.
- Train precomputed quantum kernels and evaluate accuracy.
- Save boundary plots and summary CSV files.
- Plot accuracy vs feature map for each dataset.

Key outputs:
- result/paper_result/alpha_<alpha>_<dataset>_desisionBoundaries.jpg
- result/paper_result/for_circle.csv
- result/paper_result/for_moon.csv
- result/paper_result/for_XOR.csv
- result/paper_result/<dataset>_Accuracy_vs_Feature.jpg

## Project Structure

- src/utils.py: quantum kernel training helpers.
- src/plot_utils.py: plotting helpers and output folder utilities.
- src/reproduce_utils.py: helper functions used in article reproduction notebook.
- dataset/: dataset files and metadata.
- mrk-imgs/: reference figures used in markdown cells.
- result/: generated figures and CSV outputs.

## Setup

Recommended:
- Python 3.10+
- Jupyter Notebook or JupyterLab

Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn qiskit
```

If your environment requires extra packages for circuit drawing:

```bash
pip install pylatexenc
```

## How To Run

1. Open QMLproject.ipynb and run cells top-to-bottom to generate quantum/classical comparisons for the transfusion task.
2. Open reproducingArticle.ipynb and run cells top-to-bottom to reproduce multi-dataset experiments and save CSV/plots in result/paper_result.
3. Review generated images and CSV files under result and result/paper_result.

## Notes

- Results can vary slightly with random data splits and stochastic components.
- Some output filenames contain the spelling "desision" to match current notebook save paths.
- Existing output artifacts are already present under result/paper_result and can be compared with newly generated results.

## Reference

- Article: https://www.nature.com/articles/s41598-026-39392-9
