# RFM
[KDD 2024] This is the official PyTorch implementation for the paper: "Rotative Factorization Machines"

# Guides
We reuse the baseline models and implement our models based on RecBole, an open-source, widely-used benchmark library{https://recbole.io/}.

---

# Quick Start Guide to Run RFM Model on RecBole Framework

In this document, we will provide a detailed explanation on how to run our RFM (Rotative Factorization Machines) model using the RecBole framework. Follow these steps to set up the environment, download the required datasets, and execute the model. Our model is built upon the RecBole, which is an open-source recommendation framework that facilitates the quick development and evaluation of recommendation algorithms.

### Prerequisites:

- Basic knowledge of Python programming.
- Python version 3.6 or higher installed on your machine.

### 1. Setting Up RecBole Framework:

RecBole is the base framework on which our RFM model operates. Follow the steps below to set up RecBole on your local machine.

- Navigate to [RecBole's official website](https://recbole.io/) and download the latest release of the framework.
- After downloading, extract the zip file to a suitable location on your machine.
- Now, locate the `recbole/model/context_aware_recommender` directory within the extracted folder.
- Download the `RFM.py` file provided by us and move it into the `recbole/model/context_aware_recommender` directory.

### 2. Dataset Preparation:

We employ the datasets utilized in previous works for evaluating our model. The datasets include Criteo, Avazu, ML-1M, Frappe, and ML-Tag, which are pre-processed following the methods described in EulerNet and AFN papers.

- For Criteo, Avazu and ML-1M datasets, follow the pre-processing instructions provided in the [EulerNet paper](https://dl.acm.org/doi/10.1145/3539618.3591681).
- For Frappe and ML-Tag datasets, adhere to the pre-processing guidelines stipulated in the [AFN paper](https://ojs.aaai.org/index.php/AAAI/article/view/5768).
- Ensure to place the processed datasets in a recognizable directory, as you'll need to specify the path while running the code.

### 3. Configuring and Running the RFM Model:

Now that the RecBole framework and datasets are ready, follow their instructions to run the RFM model.

- We have provided a hyper-parameter file named `hyper-para.test`. This file contains the necessary hyper-parameters for running the RFM model.


Wait for the execution to complete, and RecBole will output the results of the RFM model evaluation.

