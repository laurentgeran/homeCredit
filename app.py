#streamlit run d:/projects/openclassrooms/projets/P7_geran_laurent/homecredit/app.py
import numpy as np
import streamlit as st
import pickle
import dataAnalysis
import interpret



header1 = 'Client Selection'
header2 = 'Description or Comparison'
subheader2_1 = 'Situation at Application'
subheader2_2 = 'Previous Applications'
header3 = 'Score and interpretation'
subheader3_1 = 'Client Score'
subheader3_2 = 'Interpretation'


"""applicationInit = dataAnalysis.loadData(table='application_train', index=False)
applicationFeat = dataAnalysis.loadData(table='applicationTrain_X')
previousApplication = dataAnalysis.loadData(table='previous_application', index=False)
installmentsPayments = dataAnalysis.loadData('installments_payments', index=False)"""

"""applicationInit = dataAnalysis.loadData('data/application_train.csv')
applicationFeat = dataAnalysis.loadData('applicationTrain_X.csv')
application_y = dataAnalysis.loadData('applicationTrain_y.csv')
applicationFeatSelect = dataAnalysis.loadData('featSelectTrain_X.csv',True)
previousApplication = dataAnalysis.loadData('data/previous_application.csv')
installmentsPayments = dataAnalysis.loadData('data/installments_payments.csv')

application_Xy = applicationFeatSelect.join(application_y)"""

with open(r"clf_feat_over.pkl", "rb") as input_file:
    model = pickle.load(input_file)

st.header(header1)

SK_ID_CURR = int(st.text_input('Please enter a SK_ID_CURR', '100003'))

st.write('The entered SK_ID_CURR is', str(SK_ID_CURR))

st.header(header2)

indiv_currentInit = dataAnalysis.loadData(table='application_train',id = SK_ID_CURR, index = False)
indiv_currentFeat = dataAnalysis.loadData(table='applicationTrain_X',id = SK_ID_CURR)
indiv_currentFeatSelect = dataAnalysis.loadData(table='featSelect',id = SK_ID_CURR)
print(indiv_currentFeatSelect.columns)
indiv_currentFeatSelect_X=indiv_currentFeatSelect.drop(columns=['TARGET','SK_ID_CURR'])
#indiv_currentFeatSelect = np.array(applicationFeatSelect.iloc[indexIndiv,:]).reshape(1, -1)


ageIndiv = np.floor(-indiv_currentFeat['DAYS_BIRTH'].values[0]/365.5)
loanRateIndiv = np.round(indiv_currentFeat['LOAN_RATE'].values[0]*100,3)
telAgeIndiv = np.floor(-indiv_currentFeat['DAYS_LAST_PHONE_CHANGE'].values[0]/365.5)
previousAppIndiv = indiv_currentFeat['previousAppCounts'].values[0]
longestAppIndiv = indiv_currentFeat['CNT_PAYMENT_max'].values[0]
#longestRemainIndiv = indiv_currentFeat['CNT_INSTALMENT_FUTURE_min_max'].values[0]
maxChangeIndiv = indiv_currentFeat['NUM_INSTALMENT_VERSION_max_max'].values[0]
lastDecisionIndiv = -indiv_currentFeat['DAYS_DECISION_min'].values[0]

scoreIndiv = np.round(model.predict_proba(indiv_currentFeatSelect_X)[0][1]*100,1)
repaidStatus = indiv_currentFeatSelect.loc[:,'TARGET']

usage = st.radio(
     'What are you looking for ?',
     ('Description', 'Comparison'))
     
if usage == 'Comparison':

    criteria = st.multiselect(
    'Which criteria the subsample should have in common with the selected client ?',
    ['Gender', 'Family status'])

    if 'Gender' in criteria:
        gen = indiv_currentInit['CODE_GENDER'].values[0]
    else : 
        gen = -1

    if 'Family status' in criteria:
        fam = indiv_currentInit['NAME_FAMILY_STATUS'].values[0]
    else : 
        fam = -1

    applicationFeat = dataAnalysis.loadDataIndexes(table='applicationTrain', gender = gen, family = fam)

    print(applicationFeat)

    st.write('You selected the following similitude criteria:', criteria)

    st.subheader(subheader2_1)

    col2_1_1, col2_1_2 = st.columns(2)

    if indiv_currentFeat['x1_F'].values == 0:
        col2_1_1.metric('Gender', 'M')
    else :
        col2_1_1.metric('Gender', 'F')

    ageSample = np.floor(-applicationFeat['DAYS_BIRTH'].mean()/365.5)
    col2_1_2.metric('Age', '{age} y'.format(age = ageIndiv), delta = '{ageDelta} y'.format(ageDelta = ageIndiv- ageSample))

    col2_1_3, col2_1_4 = st.columns(2)

    loanRateSample = np.round(applicationFeat['LOAN_RATE'].mean()*100,3)
    col2_1_3.metric('Loan Rate', '{loanRate}%'.format(loanRate = loanRateIndiv), delta = '{loanRateDelta} percentage point'.format(loanRateDelta = np.round(loanRateIndiv- loanRateSample,2)))

    telAgeSample = np.floor(-applicationFeat['DAYS_LAST_PHONE_CHANGE'].mean()/365.5)
    col2_1_4.metric('Telephone Age',  '{telAge} y'.format(telAge = telAgeIndiv), delta = '{telAgeDelta} y'.format(telAgeDelta = telAgeIndiv- telAgeSample))

    st.subheader(subheader2_2)

    col2_2_1, col2_2_2, col2_2_3  = st.columns(3)

    previousAppSample = np.round(applicationFeat['previousAppCounts'].mean())
    col2_2_1.metric('Number of previous Applications', previousAppIndiv, delta = previousAppIndiv - previousAppSample)
    
    longestAppSample = np.round(applicationFeat['CNT_PAYMENT_max'].mean())
    col2_2_2.metric('Longest Application', '{longestApp} m'.format(longestApp = longestAppIndiv), delta = '{longestAppDelta} d'.format(longestAppDelta = longestAppIndiv-longestAppSample))

    longestRemainSample = np.round(applicationFeat['CNT_INSTALMENT_FUTURE_min_max'].mean())
    col2_2_3.metric('Longest Remaining Installments', longestRemainIndiv, delta = longestRemainIndiv - longestRemainSample)

    col2_2_4, col2_2_5  = st.columns(2)

    maxChangeSample = np.round(applicationFeat['NUM_INSTALMENT_VERSION_max_max'].mean())
    col2_2_4.metric('Max Changes in Installment calendar', maxChangeIndiv, delta = maxChangeIndiv - maxChangeSample)

    lastDecisionSample = np.round(-applicationFeat['DAYS_DECISION_min'].mean())
    col2_2_5.metric('Days since first decision', '{lastDecision} d'.format(lastDecision = lastDecisionIndiv), delta = '{lastDecisionDelta} d'.format(lastDecisionDelta = lastDecisionIndiv-lastDecisionSample))

    st.plotly_chart(dataAnalysis.plotHistory(SK_ID_CURR,previousApplication,installmentsPayments), use_container_width=True)

elif usage == 'Description':

    st.subheader(subheader2_1)

    col2_1_1, col2_1_2 = st.columns(2)

    if indiv_currentFeat['x1_F'].values == 0:
        col2_1_1.metric('Gender', 'M')
    else :
        col2_1_1.metric('Gender', 'F')

    col2_1_2.metric('Age', '{age} y'.format(age = ageIndiv))

    col2_1_3, col2_1_4 = st.columns(2)

    col2_1_3.metric('Loan Rate',  '{loanRate}%'.format(loanRate = loanRateIndiv))

    col2_1_4.metric('Telephone Age',  '{telAge} y'.format(telAge = telAgeIndiv))
    
    st.subheader(subheader2_2)

    col2_2_1, col2_2_2, col2_2_3  = st.columns(3)

    col2_2_1.metric('Number of previous Applications', previousAppIndiv)

    col2_2_2.metric('Longest Application', '{longestApp} m'.format(longestApp = longestAppIndiv))

    col2_2_3.metric('Longest Remaining Installments', longestRemainIndiv)

    col2_2_4, col2_2_5  = st.columns(2)

    col2_2_4.metric('Max Changes in Installment calendar', maxChangeIndiv)

    col2_2_5.metric('Days since first decision', '{lastDecision} d'.format(lastDecision = lastDecisionIndiv))

    st.plotly_chart(dataAnalysis.plotHistory(SK_ID_CURR,previousApplication,installmentsPayments), use_container_width= True)

st.header(header3)

st.subheader(subheader3_1)

col3_1, col3_2 = st.columns(2)

col3_1.metric('Score',  '{score}%'.format(score = scoreIndiv))

col3_2.metric ('Repaid status', repaidStatus)

st.subheader(subheader3_2)
"""
shapPlot, varContribNeg = interpret.shapBarPlot(model,applicationFeatSelect,indexIndiv)

st.pyplot(shapPlot)

if scoreIndiv >= 35:
    nbVar = len(varContribNeg)
    var = [var[0] for var in varContribNeg]
    nbPlot = 0
    while nbPlot < min(nbVar,3):
        st.pyplot(dataAnalysis.kde_target(var[nbPlot],application_Xy, indexIndiv))
        nbPlot+=1
"""