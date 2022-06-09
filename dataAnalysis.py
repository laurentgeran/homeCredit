import pandas as pd 
import plotly.graph_objects as go
from yaml import load

def loadData(csvName):
    return(pd.read_csv(csvName))

def mylist (df, columnFilter, columnValue, columnResult):
    return (df[df[columnFilter]==columnValue][columnResult])

def plotHistory(sk_id_curr,previousApplication,installmentsPayments):
    sk_id_prevs=previousApplication[previousApplication['SK_ID_CURR']==sk_id_curr].sort_values('DAYS_DECISION', ascending = False)
    nbCol= len(sk_id_prevs.columns)
    sk_id_prevs = sk_id_prevs.loc[(sk_id_prevs.isna().sum(axis=1)/nbCol<0.3),:]
    sk_id_prevs = sk_id_prevs['SK_ID_PREV'].values
    nbPrevApp = min(len(sk_id_prevs),3)
    fig = go.Figure()
    colors = ['green','yellow','orange']
    late = []
    for i in range(0,nbPrevApp) :
        sk_id_prev = sk_id_prevs[i]
        history = installmentsPayments[installmentsPayments['SK_ID_PREV']==sk_id_prev].sort_values("DAYS_INSTALMENT")

        fig.add_trace(
            go.Scatter(
                x=history["DAYS_INSTALMENT"].values,
                y=history["AMT_INSTALMENT"].values, 
                line = dict(color=colors[i], width=2, dash='dash'),
                mode='lines',
                name = "INSTALLMENTS_{nbInstallment}".format(nbInstallment=-i)
            )
        )

        fig.add_trace(
            go.Scatter(
                x=history["DAYS_INSTALMENT"].values,
                y=history["AMT_PAYMENT"].values,
                line=dict(color=colors[i], width=1,dash='dot'),
                mode='markers',
                name = "PAYMENTS_{nbInstallment}".format(nbInstallment=-i)
            )
        )

       
        
        for index, row in history.iterrows():
            if row["DAYS_INSTALMENT"]<row["DAYS_ENTRY_PAYMENT"]:
                fig.add_trace(
                    go.Scatter(
                        x=[row["DAYS_INSTALMENT"], row["DAYS_INSTALMENT"]],
                        y=[0, row["AMT_PAYMENT"]],
                        mode='lines',
                        line = dict(color='red', width=2),
                        showlegend=False
                    )
                )
    return(fig)