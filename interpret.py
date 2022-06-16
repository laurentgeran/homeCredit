import matplotlib.pyplot as plt
import shap


def shapBarPlot (model,df_X, index):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_X)

    varValues=[]
    for i,var in enumerate(df_X.columns):
        if shap_values[1][index][i]>0:
            varValues.append((var,shap_values[1][index][i]))
    varValues = sorted(varValues, key=lambda tup: tup[1], reverse = True)
    
    shap.bar_plot(shap_values[1][index],feature_names=df_X.columns,show=False)
    plt.title("Contributions of the 7 more impactful variables")
    barPlot = plt.gcf()
    return (barPlot,varValues)

