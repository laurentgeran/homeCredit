import matplotlib.pyplot as plt
import shap


def shapBarPlot (model,df_X, index):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_X)
    shap.bar_plot(shap_values[1][index],feature_names=df_X.columns,show=False)
    barPlot = plt.gcf
    return (barPlot)

