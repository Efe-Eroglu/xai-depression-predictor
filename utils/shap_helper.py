import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np

# 📌 1. SHAP Explainer nesnesi
def get_explainer(model):
    if isinstance(model, (RandomForestClassifier, XGBClassifier)):
        return shap.TreeExplainer(model)

    elif isinstance(model, LogisticRegression):
        return shap.LinearExplainer(model, masker=shap.maskers.Independent(np.zeros((1, model.n_features_in_))))

    elif isinstance(model, SVC):
        return shap.KernelExplainer(model.predict_proba, shap.sample_background)

    else:
        raise ValueError("Unsupported model type for SHAP explainer")

# 📌 2. SHAP değerlerini hesapla
def get_shap_values(explainer, input_df):
    """
    Tek bir örnek için SHAP değerlerini hesaplar.
    """
    return explainer(input_df)

# 📌 3. Waterfall plot
def plot_shap_waterfall(shap_values):
    """
    SHAP değerleriyle waterfall (şelale) grafiği döner.
    """
    shap_value = shap_values[0]
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_value, max_display=15, show=False)
    return fig
