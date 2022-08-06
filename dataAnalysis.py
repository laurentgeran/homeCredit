import pandas as pd 
import requests
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap

NGROK_URL = 'http://2c82-78-123-85-16.ngrok.io'

def loadData(table, id:int = -1, index = True):
    url = NGROK_URL+"/data?table="+table+"&id="+str(id)
    resp = requests.get(url)
    if index : 
        df = pd.read_json(resp.json(),orient ='records').set_index('index')
    else : 
        df = pd.read_json(resp.json(),orient ='records')
    return(df)

def loadDataCross(table, gender, family, id:int = -1, index = True):
    url = NGROK_URL+"/dataId?table="+table+"&id="+str(id)+"&gender="+str(gender)+"&family="+str(family)
    resp = requests.get(url)
    if index : 
        df = pd.read_json(resp.json(),orient ='records').set_index('index')
    else : 
        df = pd.read_json(resp.json(),orient ='records')
    return(df)

def mylist (df, columnFilter, columnValue, columnResult):
    return (df[df[columnFilter]==columnValue][columnResult])

def plotHistory(sk_id_curr,previousApplication,installmentsPayments):
    sk_id_prevs=previousApplication[previousApplication['SK_ID_CURR']==sk_id_curr].sort_values('DAYS_DECISION', ascending = False)
    nbCol= len(sk_id_prevs.columns)
    sk_id_prevs = sk_id_prevs.loc[(sk_id_prevs.isna().sum(axis=1)/nbCol<0.3),:]
    sk_id_prevs = sk_id_prevs['SK_ID_PREV'].values
    nbPrevApp = min(len(sk_id_prevs),3)
    fig = go.Figure(
        layout=go.Layout(
            title=go.layout.Title(text="History of last 3 (max) Applications")
            )
        )
    colors = ['green','yellow','orange']
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
                        name = "LATE PAYMENT",
                        showlegend=False
                    )
                )
    return(fig)

def kde_target(score, varContrib, SK_ID_CURR):

    df = loadData(table= 'featSelect',index=True)
    index = df.index[df['SK_ID_CURR'] == int(SK_ID_CURR)].values[0]

    if score >= 30:
        nbVar = len(varContrib)
        var_name = [var[0] for var in varContrib]
        nbPlot = 0

        fig, ax = plt.subplots(min(nbVar,3),1,figsize=(8,8))

        while nbPlot < min(nbVar,3):
            # Calculate the correlation coefficient between the new variable and the target
            #corr = df['TARGET'].corr(df[var_name])
            
            # Calculate medians for repaid vs not repaid
            #avg_repaid = df.loc[df['TARGET'] == 0, var_name].median()
            #avg_not_repaid = df.loc[df['TARGET'] == 1, var_name].median()
        
            repaid = df.loc[df['TARGET'] == 0, var_name[nbPlot]]
            notRepaid = df.loc[df['TARGET'] == 1, var_name[nbPlot]]

            # Group data together
            groupData = [repaid, notRepaid]
            groupLabels = ['TARGET == 0','TARGET == 1']

            
            ax[nbPlot].hist(groupData, density=True, bins=20, histtype= 'step', label=groupLabels)
            ax[nbPlot].legend(prop={'size': 10})
            ax[nbPlot].set_title('Histogram of {var}'.format(var=var_name[nbPlot]))

            ax[nbPlot].vlines(
                df.loc[:, var_name[nbPlot]][index], 0,1, transform=ax[nbPlot].get_xaxis_transform(), 
                colors='r'
                )
            nbPlot+=1
    
    # print out the correlation
    #print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr))
    # Print out average values
    #print('Median value for loan that was not repaid = %0.4f' % avg_not_repaid)
    #print('Median value for loan that was repaid =     %0.4f' % avg_repaid)

        fig.tight_layout()

    return(fig)