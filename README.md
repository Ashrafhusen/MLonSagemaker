# 📦 MLonSagemaker

This project demonstrates training and deploying a **Random Forest Classifier** using **Amazon SageMaker** for mobile price range prediction.

---

## 🔧 Project Structure

- `mob_price_classification_train.csv`: Original dataset  
- `trainV1.csv`, `testXv1.csv`: Pre-split data  
- `script.py`: SageMaker-compatible training script  
- `research.ipynb`: E2E workflow notebook  
- `requirements.txt`: Python dependencies  

---

## 🚀 Steps to Run

1. Upload files to a SageMaker Notebook instance  
2. Open and run `research.ipynb`  
3. Or train via script using SageMaker Estimator

```python
from sagemaker.sklearn.estimator import SKLearn

sk_est = SKLearn(entry_point='script.py', role='your-role',
    instance_type='ml.m5.large', framework_version='1.2-1', py_version='py3',
    hyperparameters={'n_estimators': 100, 'random_state': 0})

sk_est.fit({'train': 's3://your-bucket/trainV1.csv', 'test': 's3://your-bucket/testXv1.csv'})


Features
Data preprocessing

Model training & evaluation

Real-time endpoint deployment

Metrics: Accuracy, Precision, F1, Confusion Matrix
