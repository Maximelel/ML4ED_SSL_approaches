# Semi-Supervised Learning Approaches in Educational Contexts

Semester Project in the ML4ED (Machine Learning for Education) lab at EPFL (Sept. 2023 - Jan. 2024)

## 🌐 Overview

### 📚 Context: Reflective Writings of student teachers.

- **Educational Benefits of Reflective Writing:** Reflection on one’s performance through reflective writing offers substantial educational benefits.

- **Challenges in Reflective Writing:** Students need a very personnalized feedback from the educators and educators face complexities in teaching and assessing reflective writings.

- **ML and NLP Solutions:** ML and NLP offer potential solutions to challenges in reflective writing by automating the feedback provision.

### 🚧 Challenge

- **Labeling Hurdles in Education:** Grapple with the challenges of labeling large datasets in the educational context, demanding expert involvement and high pedagogical knowledge.

### ✨ Solution Proposed

- **Semi-Supervised Learning with Self-Training:** Explore the potential of self-learning, a semi-supervised approach that overcomes labeling challenges by leveraging small amounts of labeled data in multi-class text classification, specifically in the nuanced analysis of reflective writings.


## 🔍 Research Question

Can semi-supervised approaches reduce human effort in labeling reflective writings?

## 📊 Methodology

- **Baseline with BERT Models:** Establish baselines for multi-class and binary classification on labeled reflection datasets.

- **Learning Curves Analysis:** Explore the influence of dataset size on model performance and confidence.


---
## Reproducibility

### File structure

- **EDA_notebook** : notebook for the exploratory data analysis of the CeRED dataset.
- **notebook_kaggle** : notebook used to experiment and test the functions before running the code on the cluster
- [run_to_cluster](https://github.com/Maximelel/SP_in_ML4ED/tree/main/run_to_cluster) : directory with all the files necessary to run the code on the cluster (see [README](https://github.com/Maximelel/SP_in_ML4ED/blob/main/run_to_cluster/README.md) for more information).
- [results_visualization](https://github.com/Maximelel/SP_in_ML4ED/tree/main/results_visualization) : directory containing 2 jupyter notebooks to visualize the results (saved in the directory **results txt files**) after running the code on the cluster.
- [results txt files](https://github.com/Maximelel/SP_in_ML4ED/tree/main/results%20txt%20files) : results saved on txt files. Can be directly reused for visualization in the notebooks in **results_visualization**
  - Multi_Class_CLF
  - Multiple_Bin_CLF

### Report

The full report of this project can be found [here](https://github.com/Maximelel/SP_in_ML4ED/blob/main/report.pdf).

