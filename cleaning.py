import streamlit as st
import numpy as np
import pandas as pd

df1=df[['SEQN','RIAGENDR', 'RIDAGEYR', 'DMDEDUC2','RIDRETH3','DMDBORN4','DMDCITZN','INDFMPIR','ALQ101',
  'SIALANG', 'DMDHHSIZ', 'DMDFMSIZ','DMDHHSZA', 'DMDHHSZB', 'DMDHHSZE', 'DMDHRGND', 'DMDMARTL', 'DMDHRAGE',
  'DMDHREDU','DMDHRMAR', 'INDFMIN2','BPXPLS', 'BPXPULS', 'BPXSY2', 'BPXDI2',
  'BMXWT', 'BMXHT', 'BMXBMI', 'BMXWAIST','ALQ120Q', 'ALQ130', 'ALQ141Q', 'ALQ151',
  'BPQ020', 'BPQ030', 'BPQ080','DBQ700','DBD895', 'DBD900', 'DBD905','HUQ020', 'HUQ051',
  'MCQ010', 'MCQ080', 'MCQ220','DPQ030', 'DPQ040', 'DPQ050']]
df2=df1.loc[:,~df1.columns.duplicated()]
#dropping columns below 18
df2 = df2.drop(df2[df2.RIDAGEYR < 18].index)

df2.isnull().sum(axis = 0) #found some missing values to replace

#demographical data
demo = df2[['SEQN','RIAGENDR', 'RIDAGEYR','DMDEDUC2','RIDRETH3','SIALANG', 'DMDBORN4', 'DMDCITZN','DMDHHSIZ', 'DMDFMSIZ',
'DMDHHSZA', 'DMDHHSZB', 'DMDHHSZE', 'DMDHRGND', 'DMDHRAGE', 'DMDHREDU','DMDHRMAR', 'INDFMIN2','DMDMARTL']]
demo_df = demo.loc[:,~demo.columns.duplicated()]
demo_df['DMDBORN4'].value_counts() #check final number

# Change all "99" to 1 abd "77" to 2
demo_df.loc[demo_df['DMDBORN4'] == 99, 'DMDBORN4'] = 1
demo_df.loc[demo_df['DMDBORN4'] == 77, 'DMDBORN4'] = 2
demo_df['DMDBORN4'].value_counts()

# Citizen Status
demo_df.loc[demo_df['DMDCITZN'] == 7, 'DMDCITZN'] = 2
demo_df.loc[demo_df['DMDCITZN'] == 9, 'DMDCITZN'] = 1
demo_df['DMDCITZN'].fillna(2, inplace=True)
demo_df['DMDCITZN'].value_counts()

# Annual Family Income
demo_df.loc[demo_df['INDFMIN2'] == 77, 'INDFMIN2'] = 13
demo_df.loc[demo_df['INDFMIN2'] == 99, 'INDFMIN2'] = 12
demo_df['INDFMIN2'].fillna(12.0, inplace = True)
demo_df['INDFMIN2'].value_counts()

#Educational Level
demo_df['DMDEDUC2'].value_counts()
demo_df.loc[demo_df['DMDEDUC2'] == 7, 'DMDEDUC2'] = 4
demo_df.loc[demo_df['DMDEDUC2'] == 9, 'DMDEDUC2'] = 5
demo_df['DMDEDUC2'].fillna(4, inplace=True)
demo_df['DMDEDUC2'].value_counts()

#marital status
demo_df.loc[demo_df['DMDMARTL'] == 77, 'DMDMARTL'] = 1
demo_df.loc[demo_df['DMDMARTL'] == 99, 'DMDMARTL'] = 1
demo_df['DMDMARTL'].fillna(1,inplace=True)

dummies = ['RIAGENDR',  'RIDRETH3','DMDBORN4', 'DMDCITZN', 'SIALANG','DMDEDUC2', 'DMDHHSIZ', 'DMDFMSIZ',
              'DMDHHSZA', 'DMDHHSZB', 'DMDHHSZE', 'DMDHRGND', 'DMDHREDU', 'DMDHRMAR', 'INDFMIN2']
df_w_dummies=pd.get_dummies(demo_df, columns = dummies, drop_first = True)
final_demo=df_w_dummies.rename(columns={'RIDAGEYR': 'Age', 'RIAGENDR_2': 'Sex_Female'}) #cleaned demos with dummies

#Age
age = df[['SEQN', 'RIDAGEYR']]
age_df=age.loc[:,~age.columns.duplicated()]
age_df = age_df.rename(columns={'RIDAGEYR': 'Age'})


#Blood pressure
blood_pressure = df2[['SEQN', 'BPXPLS', 'BPXPULS', 'BPXSY2', 'BPXDI2','RIDAGEYR']]
blood_pressure = blood_pressure.rename(columns={'RIDAGEYR':'Age','BPXPLS': 'Heart Rate','BPXPULS': 'Irregular Pulse','BPXSY2': 'Systolic BP','BPXDI2': 'Diastolic BP'})
blood_pressure.isnull().sum(axis = 0) #many null values

#create a dataframe to get median value grouped by age
blood_pressure_age= blood_pressure.groupby('Age').median().reset_index()

# Fill null values from the Median
blood_pressure.loc[blood_pressure['Heart Rate'].isnull(),'Heart Rate'] = blood_pressure['Age'].map(blood_pressure_age['Heart Rate'])
blood_pressure.loc[blood_pressure['Irregular Pulse'].isnull(),'Irregular Pulse'] =blood_pressure['Age'].map(blood_pressure_age['Irregular Pulse'])
blood_pressure.loc[blood_pressure['Systolic BP'].isnull(),'Systolic BP'] = blood_pressure['Age'].map(blood_pressure_age['Systolic BP'])
blood_pressure.loc[blood_pressure['Diastolic BP'].isnull(),'Diastolic BP'] = blood_pressure['Age'].map(blood_pressure_age['Diastolic BP'])

final_bp = pd.get_dummies(blood_pressure, columns = ['Irregular Pulse'], drop_first = True)
final_bp = final_bp.rename(columns={'Irregular Pulse_2.0': 'Irregular Pulse'})
final_bp = final_bp.drop('Age', axis = 1)

#Body Measures
body_measures = df2[['SEQN', 'BMXWT', 'BMXHT', 'BMXBMI', 'BMXWAIST','RIDAGEYR']]
body_measures= body_measures.rename(columns={'RIDAGEYR':'Age','BMXWT': 'Weight (kg)',
                                                    'BMXHT' : 'Standing Height (cm)',
                                                    'BMXBMI': 'BMI',
                                                    'BMXWAIST': 'Waist Circumference (cm)'})
body_measures.isnull().sum(axis = 0) # many null values
body_measures_age = body_measures.groupby('Age').median() #grouby age to get median
body_measures.loc[body_measures['Weight (kg)'].isnull(),'Weight (kg)'] = body_measures['Age'].map(body_measures_age['Weight (kg)'])
body_measures.loc[body_measures['Standing Height (cm)'].isnull(),'Standing Height (cm)'] = body_measures['Age'].map(body_measures_age['Standing Height (cm)'])
body_measures.loc[body_measures['Waist Circumference (cm)'].isnull(),'Waist Circumference (cm)'] = body_measures['Age'].map(body_measures_age['Waist Circumference (cm)'])
body_measures['BMI'].isna().sum() #calculate missing BMI values from height and weight
body_measures['BMI'] = body_measures['BMI'].fillna(body_measures['Weight (kg)'] / (body_measures['Standing Height (cm)'] / 100)**2)
body_measures.isnull().sum(axis = 0) #zero null values
body_measures = body_measures.drop('Age', axis = 1)


#alcohol
alcohol = df2[['SEQN', 'ALQ120Q', 'ALQ130', 'ALQ141Q', 'ALQ151']]
alcohol = alcohol.loc[:,~alcohol.columns.duplicated()]

alcohol.isnull().sum(axis = 0) #too many null values
null_values=100*(alcohol.isnull().sum())/(df.shape[0])
df_null=pd.DataFrame({'percentage':null_values}) #25% of the values are null


chol = df2[['SEQN', 'BPQ020', 'BPQ030', 'BPQ080']]

#replace values
chol.loc[chol['BPQ020'] == 9, 'BPQ020'] = 0
chol.loc[chol['BPQ020'] == 2, 'BPQ020'] = 0

chol.loc[chol['BPQ030'] == 7, 'BPQ030'] = 0
chol.loc[chol['BPQ030'] == 9, 'BPQ030'] = 0
chol.loc[chol['BPQ030'] == 2, 'BPQ030'] = 0
chol['BPQ030'].fillna(0, inplace=True)
chol.loc[chol['BPQ080'] == 9, 'BPQ080'] = 0
chol.loc[chol['BPQ080'] == 2, 'BPQ080'] = 0
chol.isnull().sum(axis = 0)
#assign dummies
chol = pd.get_dummies(chol, columns =['BPQ020', 'BPQ030', 'BPQ080'], drop_first = True)

#Nutritional Diet
diet= df2[['SEQN', 'DBQ700','DBD895','DBD905']]

#replace values
diet.loc[diet['DBQ700'] == 9, 'DBQ700'] = 3
diet['DBD895'].values[diet['DBD895'].values < 1] = 0
diet.loc[diet['DBD895'] == 5555, 'DBD895'] = 22
diet.loc[diet['DBD895'] == 9999, 'DBD895'] = 0

diet['DBD905'].values[diet['DBD905'].values < 1] = 0
diet.loc[diet['DBD905'] == 6666, 'DBD905'] = 95
diet.loc[diet['DBD905'] == 7777, 'DBD905'] = 0
diet.loc[diet['DBD905'] == 9999, 'DBD905'] = 1
diet['DBD905'].fillna(0, inplace=True)
diet = pd.get_dummies(diet, columns = ['DBQ700'], drop_first = True)


#access to care
care_access = df2[['SEQN', 'HUQ020', 'HUQ051']]
#replace values
care_access.loc[care_access['HUQ020'] == 9, 'HUQ020'] = 3
care_access.loc[care_access['HUQ051'] == 77, 'HUQ051'] = 2
care_access.loc[care_access['HUQ051'] == 99, 'HUQ051'] = 2
care_access['HUQ051'].values[care_access['HUQ051'].values < 1] = 0
care_access = pd.get_dummies(care_access, columns = ['HUQ020', 'HUQ051'], drop_first = True)

#health conditions
health=df2[['SEQN', 'MCQ010', 'MCQ080', 'MCQ220']]
#replace values
health.loc[health['MCQ010'] == 2, 'MCQ010'] = 0
health.loc[health['MCQ010'] == 9, 'MCQ010'] = 0
health.loc[health['MCQ080'] == 2, 'MCQ080'] = 0
health.loc[health['MCQ080'] == 9, 'MCQ080'] = 0
health.loc[health['MCQ220'] == 2, 'MCQ220'] = 0
health.loc[health['MCQ220'] == 9, 'MCQ220'] = 0
health['MCQ220'].fillna(0, inplace=True)
health = pd.get_dummies(health, columns = ['MCQ010', 'MCQ080', 'MCQ220'], drop_first = True)


#mental health conditions
mental_health= df2[['SEQN', 'DPQ030', 'DPQ040', 'DPQ050']]
#replace values
mental_health['DPQ030'].values[mental_health['DPQ030'].values < 1] = 0
mental_health.loc[mental_health['DPQ030'] == 7, 'DPQ030'] = 2
mental_health.loc[mental_health['DPQ030'] == 9, 'DPQ030'] = 1
mental_health['DPQ030'].fillna(0, inplace=True)
mental_health['DPQ040'].values[mental_health['DPQ040'].values < 1] = 0
mental_health.loc[mental_health['DPQ040'] == 7, 'DPQ040'] = 2
mental_health['DPQ040'].fillna(0, inplace=True)
mental_health['DPQ050'].values[mental_health['DPQ050'].values < 1] = 0
mental_health.loc[mental_health['DPQ050'] == 9, 'DPQ050'] = 1
mental_health['DPQ050'].fillna(0, inplace=True)
mental_health = pd.get_dummies(mental_health, columns = ['DPQ030', 'DPQ040', 'DPQ050'], drop_first = True)

#smoking
smoking=df[['SEQN', 'SMQ040', 'SMQ020','RIDAGEYR']]
smoking_df=smoking.loc[:,~smoking.columns.duplicated()]
smoking_df = smoking_df.drop(smoking_df[smoking_df.RIDAGEYR < 18].index)


#replacing values with highest count
smoking_df.isnull().sum(axis = 0) #3534 null values
df['SMQ040'].fillna(3, inplace=True)
df.loc[df['SMQ020'] == 7, 'SMQ020'] = 1
df.loc[df['SMQ020'] == 9, 'SMQ020'] = 1
df.loc[df['SMQ020'] == 2, 'SMQ020'] = 0
#get_dummies
smoking_df = pd.get_dummies(smoking_df, columns = ['SMQ040', 'SMQ020'], drop_first = True)
smoking_df = smoking_df.drop('RIDAGEYR', axis = 1)

#diabetes
diabetes = df[['SEQN', 'DIQ010', 'DIQ160', 'DIQ170','RIDAGEYR']]
diabetes= diabetes.loc[:,~diabetes.columns.duplicated()]
diabetes = diabetes.drop(diabetes[diabetes.RIDAGEYR < 18].index)
diabetes = diabetes.rename(columns={'DIQ010': 'Diabetes', 'DIQ160': 'Prediabetes', 'DIQ170': 'At Risk','RIDAGEYR':'Age'})
diabetes.isnull().sum(axis = 0) #prediabetes and at risk have null values

#replacing values

diabetes.loc[diabetes['Diabetes'] == 9, 'Diabetes'] = 2
diabetes.loc[diabetes['Prediabetes'] == 9, 'Prediabetes'] = 2
diabetes.loc[diabetes['At Risk'] == 9, 'At Risk'] = 2
#filling NaN
diabetes_age = diabetes.groupby('Age').median() #grouby age to get median
diabetes.loc[diabetes['Prediabetes'].isnull(),'Prediabetes'] = diabetes['Age'].map(diabetes_age['Prediabetes'])
diabetes.loc[diabetes['At Risk'].isnull(),'At Risk'] = diabetes['Age'].map(diabetes_age['At Risk'])
diabetes2 = diabetes.drop('Age', axis = 1)

#st.write(diabetes2.head())
#st.write(smoking_df.head())
#st.write(mental_health.head())
#st.write(health.head())
#st.write(care_access.head())
#st.write(diet.head())
#st.write(chol.head())
#st.write(body_measures.head())
#st.write(final_bp.head())
# st.write(final_demo.head())

#create one column for diabetes

#changing diabetes column

diabetes2.loc[diabetes2['Diabetes'] == 1.0, 'Diabetes'] = 99
diabetes2.loc[diabetes2['Diabetes'] == 2.0, 'Diabetes'] = 0
diabetes2.loc[diabetes2['Diabetes'] == 3.0, 'Diabetes'] = 10
diabetes2.loc[diabetes2['Diabetes'] == 9.0, 'Diabetes'] = 0

#changing prediabetes column

diabetes2['Prediabetes'].fillna(0, inplace = True)
diabetes2.loc[diabetes2['Prediabetes'] == 9, 'Prediabetes'] = 0
diabetes2.loc[diabetes2['Prediabetes'] == 2, 'Prediabetes'] = 0
diabetes2.loc[diabetes2['Prediabetes'] == 1, 'Prediabetes'] = 5

#changing at risk column

diabetes2['At Risk'].fillna(0, inplace = True)
diabetes2.loc[diabetes2['At Risk'] == 9, 'At Risk'] = 0
diabetes2.loc[diabetes2['At Risk'] == 2, 'At Risk'] = 0
diabetes2.loc[diabetes2['At Risk'] == 1, 'At Risk'] = 3

diabetes2['Combined'] = diabetes2['Diabetes'] + diabetes2['Prediabetes'] + diabetes2['At Risk']


diabetes2.loc[diabetes2['Combined'] == 99, 'Combined'] = 1
diabetes2.loc[diabetes2['Combined'] == 13, 'Combined'] = 2
diabetes2.loc[diabetes2['Combined'] == 10, 'Combined'] = 2
diabetes2.loc[diabetes2['Combined'] == 8, 'Combined'] = 2
diabetes2.loc[diabetes2['Combined'] == 5, 'Combined'] = 2
diabetes2.loc[diabetes2['Combined'] == 3, 'Combined'] = 2
diabetes2.loc[diabetes2['Combined'] == 0, 'Combined'] = 3


diabetes_col=diabetes2[['SEQN','Combined']]

#1 = Diabetic
#2 = Prediabetic
#3 = Healthy

final_df=pd.concat([diabetes_col, body_measures, final_bp, smoking_df, mental_health, health, care_access, diet, chol,final_demo], axis=1)
final_df= final_df.loc[:,~final_df.columns.duplicated()]
#final=to_csv('final.csv')
