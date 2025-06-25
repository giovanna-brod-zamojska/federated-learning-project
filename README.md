# federated-learning-project

# Notebooks

All our experiments were carried out on Google Colab using the following notebooks:

- src/notebooks/centralized-baseline-training.ipynb: notebook for training and evaluating the Centralized Baseline model

- src/notebooks/centralized-baseline-model-editing.ipynb: notebook for training and evaluating the Centralized Baseline model when sparse finetuning is applied using a mask calibration strategy that considers as trainable a subset of the k least important parameters, importance computed using an approximated version of the Fisher Information Matrix (FIM)

- src/notebooks/Federated_Learning_decentralized_baseline.ipynb: notebook for training and evaluating the Federated Averaged  models, under both IID and NON-IID settings, when sparse finetuning is applied using a mask calibration strategy that considers as trainable a subset of the k least important parameters, importance computed using an approximated version of the Fisher Information Matrix (FIM)

- src/notebooks/personal_contribution_iid.ipynb: notebook for training and evaluating the Federated Averaged model under IID setting, when sparse finetuning is applied considering different mask calibration strategies 

- src/notebooks/personal_contribution_non_iid.ipynb: notebook for training and evaluating the Federated Averaged model under NON-IID setting, when sparse finetuning is applied considering different mask calibration strategies 
