import shap
import matplotlib.pyplot as plt

# 📌 1. SHAP Explainer nesnesi
def get_explainer(model):
    """
    Modele uygun SHAP explainer nesnesini döndürür.
    """
    if hasattr(model, "predict_proba") and "tree" in str(type(model)).lower():
        return shap.TreeExplainer(model)
    else:
        return shap.Explainer(model)

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
