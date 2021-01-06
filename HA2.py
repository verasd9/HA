import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import date
import plotly as py
from datetime import datetime, timedelta
import datetime as dt
from matplotlib.figure import Figure
import warnings
warnings.filterwarnings('ignore')

#loading the data
@st.cache(allow_output_mutation=True)
def get_data1():
    df_demo=pd.read_csv('demographic.csv')
    return df_demo
@st.cache(allow_output_mutation=True)
def get_data2():
    df_diet=pd.read_csv('diet.csv')
    return df_diet
@st.cache(allow_output_mutation=True)
def get_data3():
    df_exam=pd.read_csv('examination.csv')
    return df_exam
@st.cache(allow_output_mutation=True)
def get_data4():
    df_lab=pd.read_csv('labs.csv')
    return df_lab
@st.cache(allow_output_mutation=True)
def get_data5():
    df_med=pd.read_csv('medications.csv',encoding= 'unicode_escape')
    return df_med
@st.cache(allow_output_mutation=True)
def get_data6():
    df_ques=pd.read_csv('questionnaire.csv')
    return df_ques

@st.cache(allow_output_mutation=True)
def get_data7():
    final=pd.read_csv('data.csv')
    return final


df_demo=get_data1()
df_diet=get_data2()
df_exam=get_data3()
df_lab=get_data4()
df_med=get_data5()
df_ques=get_data6()
final=get_data7()

df=pd.concat([df_demo, df_diet, df_exam, df_lab, df_ques], axis=1)



st.markdown(
    """
<style>
.reportview-container .markdown-text-container {
    font-family: sans-serif;
}
.sidebar .sidebar-content {
    background-image: linear-gradient(#979594,#922B21);
    color: white;
}
.Widget>label {
    color: black;
    font-family: sans-serif;
}
[class^="st-b"]  {
    color: black;
    font-family: sans-serif;
}
.st-bb {
    background-color: transparent;
}
.st-at {
    background-color: #C5C3C3;
}
footer {
    font-family: sans-serif;
}
.reportview-container .main footer, .reportview-container .main footer a {
    color: #0c0080;
}
header .decoration {
    background-image: none;
}

</style>
""",
    unsafe_allow_html=True,
)
st.sidebar.markdown("## Navigation")
st.sidebar.markdown("① ** Data Exploration **")
pages=st.sidebar.selectbox("Explore by",
        ["Topics","Demographics","Lifestyle","Diseases"])

if pages=="Topics":
    st.title("National Health and Nutrition Survey")

    from pathlib import Path
    def read_markdown_file(markdown_file):
        return Path(markdown_file).read_text()
    intro_markdown = read_markdown_file("intro1.md")
    st.markdown(intro_markdown, unsafe_allow_html=True)
    st.markdown("---")

    st.sidebar.markdown("② ** Predictive Analysis **")
    pages2=st.sidebar.selectbox("Predict",
            ["Topics","Diabetes Prediction","Disease Prediction"])
    st.sidebar.markdown("---")
    st.sidebar.markdown("ℹ️ ** Details **")
    desc_check = st.sidebar.checkbox("📃 Dataset Description")
    #desc_markdown = read_markdown_file("data_description.md")
    dict_check = st.sidebar.checkbox("📕 Data Dictionary")
    #dict_markdown = read_markdown_file("data_dictionary.md")

elif pages=="Demographics":
    demo1=pd.read_csv("demo.csv")
    demo1.info()
    st.title("Demographics")

    st.set_option('deprecation.showPyplotGlobalUse', False)
    def Age_Categories(row):
        if row['age_years']in range(0,18):
            return "0.5-20"
        elif row['age_years'] in range(18,25):
            return "18-25"
        elif row['age_years']in range(25,45):
            return "25-45"
        elif row['age_years'] in range (45,60):
            return "45-60"
        else:
            return "Above 80"
    demo1['Age_Categories']=demo1.apply(lambda row: Age_Categories(row),axis=1)

    def Education(row):
        if (row['degree_2']==1):
            return "Less than 9th grade"
        elif (row['degree_2']==2):
            return "9-11th grade"
        elif (row['degree_2']==3):
            return "High school graduate"
        elif (row['degree_2']==4):
            return "Some college or AA degree"
        else:
            return "College graduate or above"

    demo1['Education']=demo1.apply(lambda row: Education(row),axis=1)
    demo2=demo1[demo1.Education != 'Don\'t Know']
    #household income level

    def Income(row):
        if (row['household_income']==1):
            return "$ 0 to $ 4,999"
        elif (row['household_income']==2):
            return "$ 5,000 to $ 9,999"
        elif (row['household_income']==3):
            return "$10,000 to $14,999"
        elif (row['household_income']==4):
            return "$15,000 to $19,999"
        elif (row['household_income']==5):
            return "$20,000 to $24,999"
        elif (row['household_income']==6):
            return "$25,000 to $34,999"
        elif (row['household_income']==7):
            return "$35,000 to $44,999"
        elif (row['household_income']==8):
            return "$45,000 to $54,999"
        elif (row['household_income']==9):
            return "$55,000 to $64,999"
        elif (row['household_income']==10):
            return "$65,000 to $74,999"
        elif (row['household_income']==14):
            return "$75,000 to $99,999"
        elif (row['household_income']==12):
            return "$20,000 and Over"
        elif (row['household_income']==13):
            return "Under $20,000"
        else:
            return "$100,000 and Over"


    demo1['Income']=demo1.apply(lambda row: Income(row),axis=1)

    #species = st.multiselect('Show data per variety?', demo1['Age_Categories'].unique())
    col1 = st.selectbox('Which feature on x?', ['Gender','Race','Education'])
    col2 = st.selectbox('Which feature on y?', [demo1.columns[3:36]])
    if col1=="Gender":
        if col2=="age_years":
            demo1.groupby(['Age_Categories','gender']).size().unstack().plot(kind='bar',stacked=True,color=('darkred','silver','darkgrey'))
            st.pyplot()
        if col2=="marital_status":
            demo1.groupby(['marital_status','gender']).size().unstack().plot(kind='bar',stacked=True,color=('darkred','silver','darkgrey'))
            st.pyplot()
        if col2=="degree_2":
            demo1.groupby(['Education','gender']).size().unstack().plot(kind='bar',stacked=True,color=('darkred','silver','darkgrey'))
            st.pyplot()
        if col2=="country_birth":
            demo1.groupby(['country_birth','gender']).size().unstack().plot(kind='bar',stacked=True,color=('darkred','silver','darkgrey'))
            st.pyplot()
        if col2=="pregnancy":
            demo1.groupby(['pregnancy','gender']).size().unstack().plot(kind='bar',stacked=True,color=('darkred','silver','darkgrey'))
            st.pyplot()
        if col2=="household_income":
            demo1.groupby(['household_income','gender']).size().unstack().plot(kind='bar',stacked=True,color=('darkred','silver','darkgrey'))
            st.pyplot()
    if col1=="Race":
        if col2=="age_years":
            demo1.groupby(['race','Age_Categories']).size().unstack().plot(kind='bar',stacked=True,color=('darkred','indianred','silver','darkgrey','dimgrey'))
            st.pyplot()
        if col2=="gender":
            demo1.groupby(['race','gender']).size().unstack().plot(kind='bar',stacked=True,color=('darkred','silver','darkgrey','dimgrey'))
            st.pyplot()
        if col2=="US_citizen":
            demo1.groupby(['race','US_citizen']).size().unstack().plot(kind='bar',stacked=True,color=('darkred','silver','darkgrey','dimgrey'))
            st.pyplot()
        if col2=="country_birth":
            demo1.groupby(['race','country_birth']).size().unstack().plot(kind='bar',stacked=True,color=('darkred','silver','darkgrey','dimgrey'))
            st.pyplot()
    if col1=="Education":
        if col2=="gender":
            demo1.groupby(['degree_2','gender']).size().unstack().plot(kind='bar',stacked=True,color=('darkred','silver','darkgrey','dimgrey'))
            st.pyplot()
        if col2=="household_income":
            demo1.groupby(['Income','Education']).size().unstack().plot(kind='bar',stacked=True,color=('darkred','silver','darkgrey','dimgrey'))
            st.pyplot()

    #new_df = df[(df['variety'].isin(species))]
    #st.write(new_df)
    #fig = px.scatter(new_df, x =col1,y=col2, color='variety')
    #t.plotly_chart(fig)

elif pages=="General Health Status":
    st.write("General Health")
    exam=pd.read_csv('exam.csv')
    #Obesity
    exam['BMI'] = exam['BMI'].fillna(exam['weight_kg'] / (exam['height_cm'] / 100)**2)

elif pages=="Lifestyle":
    st.write("Lifestyle")
    radio=st.sidebar.radio("Choose",("Alcohol Consumption","Occupation","Physical Activity","Smoking Habits"))
    if radio=="Alcohol Consumption":
        #alcohol consumption
        alc2=df_ques[['ALQ130']].dropna()
        alc2=alc2['ALQ130'].value_counts().reset_index()
        st.subheader("Avg # alcoholic drinks/day - past 12 mos")
        fig = px.scatter(alc2, x="index", y="ALQ130",size="ALQ130")
        fig.update_layout(plot_bgcolor='white')
        st.plotly_chart(fig)
    elif radio=="Occupation":
        occ=df_ques[['OCQ180']].dropna()
        occ=occ['OCQ180'].value_counts().reset_index()
        mean=occ['index'].mean()
        max=occ['index'].max()
        min=occ['index'].min()
        st.write("Maximum hours spent:", max, 'hours')
        st.write("Minimum hours spent:", min, 'hours')

        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = mean,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Average hours worked during the week"},
            gauge = {'bar': {'color': "firebrick"},
                #'axis': {'range': [None, 5]},
                     'steps' : [
                         {'range': [0, 2.5], 'color': "lightgray"},
                         {'range': [2.5, 4], 'color': "gray"}],
                    }))
        st.plotly_chart(fig)
    elif radio=="Physical Activity":
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.header("Work Activity and physical exercise")
        act=df_ques[['PAQ605','PAQ620','PAQ635']].dropna()
        act.rename(columns={'PAQ605': 'Vigorous Work Activity', 'PAQ620': 'Moderate Work Activity','PAQ635':'Walking/Biking'}, inplace=True)
        act.groupby(['Vigorous Work Activity','Walking/Biking']).size().unstack().plot(kind='bar',stacked=True,color=('darkred','silver','darkgrey'))
        st.pyplot()

    elif radio=="Smoking Habits":
        st.title("Smoking Habits")
        cig=st.selectbox('Choose Target',["Adults","Teenagers"])
        if cig=="Adults":
            ad=df_ques['SMQ020'].dropna()
            st.write(ad)
        elif cig=="Teenagers":
            teen=df_ques['SMQ621'].dropna()
            st.write(teen)

elif pages=='Diseases':
    st.write("Diseases")

if pages2=="Diabetes Prediction":
    final=pd.read_csv("data.csv")
    explore=st.checkbox('Target Exploration')
    if explore:
        st.subheader('Target: Diabetes')
        st.write('Risk Factors: Insulin, Glucose, HDL, Total Cholesterol, Examination Data, Body Measurements, Blood Preasure, Diet Interview, Nutrition, Demographics, Questionnaire Data, Alcohol, Smoking')
        st.set_option('deprecation.showPyplotGlobalUse', False)

        target=final['Diabetes'].value_counts().reset_index()
        target.rename(columns={'index': 'Status'}, inplace=True)
        def Type(row):
            if (row['Status']==1):
                return "Diabetic"
            elif (row['Status']==2):
                return "Pre-Diabetic"
            else:
                return "Non Diabetic"
        target['Type']=target.apply(lambda row: Type(row),axis=1)
        st.write(target)

        import matplotlib.ticker as mtick

        plt.bar(target['Type'], target['Diabetes'], color ='firebrick',
                width = 0.4)

        plt.xlabel("Diabetes")
        plt.ylabel("Values")

        st.pyplot()


    machinelearn=st.checkbox('Machine Learning')
    if machinelearn:
        alg = ['Random Forest', 'Support Vector Machine','KNN']
        classifier = st.selectbox('Which algorithm?', alg)
        if classifier=='Random Forest':
            #splitting the dataset
            X = final.iloc[:,1:]
            y = final['Diabetes']
            import sklearn
            from sklearn import metrics
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

            #over_sampling
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=2)
            X_train, y_train = sm.fit_sample(X_train, y_train.ravel())


            from sklearn import metrics
            from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(bootstrap = False,
                                         criterion =  'entropy',
                                         max_depth = None,
                                         max_features = 'auto',
                                         min_samples_leaf = 2,
                                         min_samples_split = 9,
                                         n_estimators = 115,
                                         random_state = 47)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write(metrics.accuracy_score(y_test, y_pred))

            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

            st.write('Accuracy Score: ', accuracy_score(y_test, y_pred))
            st.write('F1 Score:', f1_score(y_test, y_pred, average="macro"))
            st.write('Precision Score:', precision_score(y_test, y_pred, average="macro"))
            st.write('Recall Score:',recall_score(y_test, y_pred, average="macro"))
            report = classification_report(y_test, y_pred, labels=None,
                                           target_names=['Diabetic', 'Prediabetic', 'Healthy'],
                                           sample_weight=None, digits=2, output_dict=False)

            #st.table(report)

            from sklearn.metrics import confusion_matrix
            import seaborn as sns
            import matplotlib.pyplot as plt


            cm = confusion_matrix(y_test, y_pred)
            ax= plt.subplot()
            fig = sns.heatmap(cm, annot=True, ax = ax, annot_kws={"size": 18}, fmt="d", cmap="YlGnBu")

            ax.set_xlabel('Predicted Labels', fontsize = 20)
            ax.set_ylabel('True Labels', fontsize = 20)
            ax.set_title('Confusion Matrix', fontsize = 20)
            ax.yaxis.set_ticklabels([' Healthy', 'Prediabetic', 'Diabetic'], fontsize = 13)
            ax.xaxis.set_ticklabels(['Diabetic', 'Prediabetic', 'Healthy'], fontsize = 13)
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)

            from sklearn.tree import export_graphviz

            # Create a list of features
            features = list(final.columns.values)
            features = features[1:]

            # Extract single tree
            estimator = model.estimators_[114]

            feature_importances = pd.DataFrame(model.feature_importances_,
                                   index = features,
                                   columns=['Importance']).sort_values('Importance', ascending=False)
            st.write(feature_importances)

            from sklearn.metrics import roc_curve, auc
            from scipy import interp
            from itertools import cycle

            n_classes = 3
            ### MACRO
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_test))[:, i], np.array(pd.get_dummies(y_pred))[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])


            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            lw=2
            plt.figure(figsize=(10,8))
            plt.plot(fpr["macro"], tpr["macro"],
                     label='macro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["macro"]),
                     color='green', linestyle=':', linewidth=4)

            colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                         label='ROC curve of class {0} (area = {1:0.2f})'
                         ''.format(i, roc_auc[i]))

            plt.plot([0, 1], [0, 1], 'k--',color='red', lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.annotate('Random Guess',(.5,.48),color='red', fontsize = 15)
            plt.xlabel('False Positive Rate', fontsize = 20)
            plt.ylabel('True Positive Rate', fontsize = 20)
            plt.title('ROC Curve', fontsize = 24)
            plt.legend(loc="lower right", prop={'size': 17})
            plt.show()


        elif classifier=="Support Vector Machine":
            #splitting the dataset
            X = final.iloc[:,1:]
            y = final['Diabetes']
            import sklearn
            from sklearn import metrics
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

            #over_sampling
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=2)
            X_train, y_train = sm.fit_sample(X_train, y_train.ravel())
            from sklearn.svm import SVC

            from sklearn.metrics import confusion_matrix


            model = SVC(       C = 1,
                       coef0 = 0.0,
                       decision_function_shape = 'ovr',
                       degree = 3,
                       gamma = 0.001,
                       kernel = 'rbf',
                       max_iter = 1,
                       probability = False,
                       random_state = 47,
                       shrinking = True,
                       tol = 0.001,
                       verbose = 2
               )
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc2= model.score(X_test, y_test)
            st.write(metrics.accuracy_score(y_test, y_pred))
            #st.write('Accuracy:', acc2)

            st.write('Accuracy Score: ', accuracy_score(y_test, y_pred))
            st.write('F1 Score:', f1_score(y_test, y_pred, average="macro"))
            st.write('Precision Score:', precision_score(y_test, y_pred, average="macro"))
            st.write('Recall Score:',recall_score(y_test, y_pred, average="macro"))
            report2 = classification_report(y_test, y_pred, labels=None,
                                           target_names=['Diabetic', 'Prediabetic', 'Healthy'],
                                           sample_weight=None, digits=2, output_dict=False)


            cm = confusion_matrix(y_test, y_pred)
            ax= plt.subplot()
            fig = sns.heatmap(cm, annot=True, ax = ax, annot_kws={"size": 18}, fmt="d", cmap="YlGnBu")

            ax.set_xlabel('Predicted Labels', fontsize = 20)
            ax.set_ylabel('True Labels', fontsize = 20)
            ax.set_title('Confusion Matrix', fontsize = 20)
            ax.yaxis.set_ticklabels([' Healthy', 'Prediabetic', 'Diabetic'], fontsize = 13)
            ax.xaxis.set_ticklabels(['Diabetic', 'Prediabetic', 'Healthy'], fontsize = 13)
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)

        elif classifier=="KNN":
            #splitting the dataset
            X = final.iloc[:,1:]
            y = final['Diabetes']
            import sklearn
            from sklearn import metrics
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

            #over_sampling
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=2)
            X_train, y_train = sm.fit_sample(X_train, y_train.ravel())

            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

            model = KNeighborsClassifier()
            model.fit(X_train, y_train)
            test_preds = model.predict(X_test)

            st.write('Accuracy Score: ', accuracy_score(y_test, test_preds))
            st.write('F1 Score:', f1_score(y_test, test_preds, average="macro"))
            st.write('Precision Score:', precision_score(y_test, test_preds, average="macro"))
            st.write('Recall Score:',recall_score(y_test, test_preds, average="macro"))

            cm = confusion_matrix(y_test, test_preds)
            ax= plt.subplot()
            fig = sns.heatmap(cm, annot=True, ax = ax, annot_kws={"size": 18}, fmt="d", cmap="YlGnBu")

            ax.set_xlabel('Predicted Labels', fontsize = 20)
            ax.set_ylabel('True Labels', fontsize = 20)
            ax.set_title('Confusion Matrix', fontsize = 20)
            ax.yaxis.set_ticklabels([' Healthy', 'Prediabetic', 'Diabetic'], fontsize = 13)
            ax.xaxis.set_ticklabels(['Diabetic', 'Prediabetic', 'Healthy'], fontsize = 13)
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)

elif pages2=="Prediction":
    st.write("Heart Disease")
    #splitting the dataset
    X = final.iloc[:,1:]
    y = final['Diabetes']
    import sklearn
    from sklearn import metrics
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    #over_sampling
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=2)
    X_train, y_train = sm.fit_sample(X_train, y_train.ravel())
    from sklearn.cluster import KMeans
    kms = KMeans(n_clusters = 3, tol = 0.0005, algorithm="auto")

    kms.fit_predict(X_train)

    st.write ("parameters: ", kms.get_params)
    st.write ("predict: ", kms.predict)
    st.write ("\nscore: %.2f" % kms.score(X_test))
