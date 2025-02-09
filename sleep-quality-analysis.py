# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 13:22:45 2025

@author: stesc
"""

#importing all libraries and functions needed
import pandas as pd
from pandas import read_excel
import scipy.stats as stt
import fredapi as fa
from scipy.stats import mode
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import normaltest
from scipy.stats import kstest
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import scipy.stats as statistica
import scipy.stats
from scipy import stats
from scipy.stats import levene
from statsmodels.formula.api import ols
import statsmodels.api as sm
import numpy as np
import scipy as sp
import scipy.stats
from pandas import read_csv
from datetime import datetime
from scipy.stats import chi2_contingency
from scipy.stats import chisquare
from scipy.stats import spearmanr


#2 Preliminary operations
#import the data from the CSV file
sleep_data = read_csv("Sleep_health_and_lifestyle_dataset.csv")
sleep_data.columns = sleep_data.columns.str.replace(' ', '.', regex=False)

#create a new data set with the following conditions
sleep_quality_check = sleep_data[
    ((sleep_data['Sleep.Duration'] >= 5.0) & (sleep_data['Sleep.Duration'] <= 7.5)) &
    ((sleep_data['Stress.Level'] >= 2) & (sleep_data['Stress.Level'] <= 9)) &
    ((sleep_data['Quality.of.Sleep'] >= 3) & (sleep_data['Quality.of.Sleep'] <= 9))]

#drop the unnecessary columns
drop_columns = ['Daily.Steps', 'Blood.Pressure']
sleep_quality_check = sleep_quality_check.drop(columns = drop_columns)
#fill the NA values with the new value 'None'
sleep_quality_check = sleep_quality_check.fillna({'Sleep.Disorder': 'None'})

#create a new CSV file with the data that meet the conditions above
sleep_quality_check.to_csv('sleep_quality_check.csv', index = False)

#shows the data types for each variable included in the data frame
print(sleep_quality_check.dtypes)

#transform the Sleep.Disorder variable to categorical
sleep_quality_check['Sleep.Disorder'] = pd.Categorical(
    sleep_quality_check['Sleep.Disorder'],
    categories = ['Insomnia', 'None', 'Sleep Apnea'])

#create a new categorical variable with 3 groups based on the intervals
sleep_quality_check['sleep_disorder_stress'] = pd.cut(
    sleep_quality_check['Stress.Level'], bins = [0, 3, 5, 9],
    labels = ['Low', 'Moderate', 'High'],
    right = True)

#print(sleep_quality_check)
#result = sleep_quality_check[sleep_quality_check['sleep_disorder_stress'] == 'Low']
#print(result)

#shows the dimension of the data frame
print(sleep_quality_check.shape)
#shows the structure of the data
print(sleep_quality_check.info())
#show the names of columns
print(sleep_quality_check.columns)

#
#print(sleep_quality_check.head)
#see the unique values for each variable below
print(sleep_quality_check['Sleep.Disorder'].unique())
print(sleep_quality_check['sleep_disorder_stress'].unique())

#2.1
#create a subset data frame with only the numerical variables
data_sep_num = sleep_quality_check.loc[:, ['Age', 'Sleep.Duration', 'Stress.Level',
                                       'Quality.of.Sleep', 'Physical.Activity.Level',
                                       'Heart.Rate']]

#set the option to show all variables in the console
pd.set_option('display.max_columns', None)
#show the statistical summary of the new data frame
summary = data_sep_num.describe()
print(summary)

#show the statistical summary for each variable
sleep_quality_check['Age'].describe()
sleep_quality_check['Sleep.Duration'].describe()
sleep_quality_check['Stress.Level'].describe()
sleep_quality_check['Quality.of.Sleep'].describe()
sleep_quality_check['Physical.Activity.Level'].describe()
sleep_quality_check['Heart.Rate'].describe()

#skew & kurtosis
print('The skewness of Age is: ', skew(sleep_quality_check['Age']))
print('The kurtosis of Age is: ', kurtosis(sleep_quality_check['Age']))

print('The skewness of Sleep.Duration is: ', skew(sleep_quality_check['Sleep.Duration']))
print('The kurtosis of Sleep.Duration is: ', kurtosis(sleep_quality_check['Sleep.Duration']))

print('The skewness of Stress.Level is: ', skew(sleep_quality_check['Stress.Level']))
print('The kurtosis of Stress.Level is: ', kurtosis(sleep_quality_check['Stress.Level']))

print('The skewness of Quality.of.Sleep is: ', skew(sleep_quality_check['Quality.of.Sleep']))
print('The kurtosis of Quality.of.Sleep is: ', kurtosis(sleep_quality_check['Quality.of.Sleep']))

print('The skewness of Physical.Activity.Level is: ', skew(sleep_quality_check['Physical.Activity.Level']))
print('The kurtosis of Physical.Activity.Level is: ', kurtosis(sleep_quality_check['Physical.Activity.Level']))

print('The skewness of Heart.Rate is: ', skew(sleep_quality_check['Heart.Rate']))
print('The kurtosis of Heart.Rate is: ', kurtosis(sleep_quality_check['Heart.Rate']))

#non numerice variables
#data_groups = sleep_quality_check.groupby('sleep_disorder_stress')['Sleep.Duration'].describe()
#data_groups = data_groups.round(4)

sleep_quality_check['sleep_disorder_stress'].describe()
sleep_quality_check['Sleep.Disorder'].describe()

#2.2 visual analysis
#create a histogram for each variable
#hist1 Age
plt.figure(figsize=(10,8))
plt.hist(x= sleep_quality_check['Age'],
         edgecolor='white',
         color = 'green')
plt.title('Histogram of sleep_quality_check["Age"]')
plt.xlabel('Age of people in the study')
plt.ylabel('Frequency')


#hist2 Sleep Duration
plt.figure(figsize=(10,8))
plt.hist(x= sleep_quality_check['Sleep.Duration'],
         edgecolor='white',
         color = 'green')
plt.title('Histogram of sleep_quality_check["Sleep.Duration"]')
plt.xlabel('Sleep Duration per night')
plt.ylabel('Frequency')

#hist3 Stress Level
plt.figure(figsize=(10,8))
plt.hist(x= sleep_quality_check['Stress.Level'],
         edgecolor='white',
         color = 'green')
plt.title('Histogram of sleep_quality_check["Stress.Level"]')
plt.xlabel('Stress Level')
plt.ylabel('Frequency')

#hist4 Quality of Sleep
plt.figure(figsize=(10,8))
plt.hist(x= sleep_quality_check['Quality.of.Sleep'],
         edgecolor='white',
         color = 'green')
plt.title('Histogram of sleep_quality_check["Quality.of.Sleep"]')
plt.xlabel('Quality of Sleep')
plt.ylabel('Frequency')

#hist5 Physical Activity Level
plt.figure(figsize=(10,8))
plt.hist(x= sleep_quality_check['Physical.Activity.Level'],
         edgecolor='white',
         color = 'green')
plt.title('Histogram of sleep_quality_check["Physical.Activity.Level"]')
plt.xlabel('Physical Activity Level')
plt.ylabel('Frequency')

#hist6 Heart Rate
plt.figure(figsize=(10,8))
plt.hist(x= sleep_quality_check['Heart.Rate'],
         edgecolor='white',
         color = 'green')
plt.title('Histogram of sleep_quality_check["Heart.Rate"]')
plt.xlabel('Heart Rate')
plt.ylabel('Frequency')

#the visual analysis of non numeric variables
#create a plot for each variable
#plot1
plt.figure(figsize=(10,8))
plt.hist(x= sleep_quality_check['Sleep.Disorder'],
         edgecolor='white',
         color = 'red')
plt.title('Tulburari de somn intalnite')
plt.xlabel('Sleep Disorder')
plt.ylabel('Frequency')

#plot2
plt.figure(figsize=(10,8))
plt.hist(x= sleep_quality_check['sleep_disorder_stress'],
         edgecolor='white',
         color = 'blue')
plt.title('Nivelul de stres in functie de probleme preexistente')
plt.xlabel('Stress Level')
plt.ylabel('Frequency')

#2.3
#identify the outliers using box plots

#bplot1
plt.figure(figsize=(6,7))
plt.boxplot(sleep_quality_check['Age'])
plt.title('Boxplot of sleep_quality_check["Age"]')

#bplot2
plt.figure(figsize=(6,7))
plt.boxplot(sleep_quality_check['Sleep.Duration'])
plt.title('Boxplot of sleep_quality_check["Sleep.Duration"]')

#bplot3
plt.figure(figsize=(6,7))
plt.boxplot(sleep_quality_check['Stress.Level'])
plt.title('Boxplot of sleep_quality_check["Stress.Level"]')

#bplot4
plt.figure(figsize=(6,7))
plt.boxplot(sleep_quality_check['Quality.of.Sleep'])
plt.title('Boxplot of sleep_quality_check["Quality.of.Sleep"]')

#bplot5
plt.figure(figsize=(6,7))
plt.boxplot(sleep_quality_check['Physical.Activity.Level'])
plt.title('Boxplot of sleep_quality_check["Physical.Activity.Level"]')

#bplot6
plt.figure(figsize=(6,7))
plt.boxplot(sleep_quality_check['Heart.Rate'])
plt.title('Boxplot of sleep_quality_check["Heart.Rate"]')


#cap3
#create a new table to find out the frequencies
tabelarea_datelor = pd.crosstab(sleep_quality_check['Sleep.Disorder'],
                                sleep_quality_check['sleep_disorder_stress'])

print(tabelarea_datelor)

#partial relative frequencies
frecvente_relative_partiale = pd.crosstab(
    sleep_quality_check['Sleep.Disorder'], 
    sleep_quality_check['sleep_disorder_stress'], 
    normalize=True,
    margins=False)

print(frecvente_relative_partiale)

#relatively conditioned frequencies - Sleep.Disorder (line)
frecvente_relativ_conditionate_SD = pd.crosstab(
    sleep_quality_check['Sleep.Disorder'], 
    sleep_quality_check['sleep_disorder_stress'], 
    normalize='index',
    margins=False)

print(frecvente_relativ_conditionate_SD)

#conditional relative frequencies - sleep_disorder_stress (column)
frecvente_relativ_conditionate_SDS = pd.crosstab(
    sleep_quality_check['Sleep.Disorder'], 
    sleep_quality_check['sleep_disorder_stress'], 
    normalize='columns',
    margins=False)

print(frecvente_relativ_conditionate_SDS)

#relatively marginal frequencies
frecvente_relative_marginale = pd.crosstab(
    sleep_quality_check['Sleep.Disorder'], 
    sleep_quality_check['sleep_disorder_stress'], 
    normalize=True,
    margins=True)

print(frecvente_relative_marginale)

#3.2
#conducting a Chi-square test on the table above

chisq, df, p_value, _ = chi2_contingency(tabelarea_datelor)

chi2_contingency(tabelarea_datelor)

print('Chisq: ', chisq)
print('df: ', df)
print('P-value: ', p_value)

#3.3
tabel_frecvente_SDS = sleep_quality_check['sleep_disorder_stress'].value_counts()

X_sq, p_value = chisquare(tabel_frecvente_SDS)

print('X-sq: ', X_sq)
print('df: ', len(tabel_frecvente_SDS) - 1)
print('P-value: ', p_value)


#concordance analysis for the sleep_disorder_stress variable and a theoretical distribution
distributie_teoretica = [0.1, 0.3, 0.6]

tabel_frecvente_SDS2 = sleep_quality_check['sleep_disorder_stress'].value_counts()

distributie_frecv_asteptate = [prob * len(sleep_quality_check)
                               for prob in distributie_teoretica]

X_sq, p_value = chisquare(tabel_frecvente_SDS2, distributie_frecv_asteptate)

print('X-sq: ', X_sq)
print('df: ', len(tabel_frecvente_SDS2) - 1)
print('P-value: ', p_value)

#analysis of the 2nd variable Sleep.Disorder
tabel_frecvente_SleepD = sleep_quality_check['Sleep.Disorder'].value_counts()

X_sq, p_value = chisquare(tabel_frecvente_SleepD)

print('X-sq: ', X_sq)
print('df: ', len(tabel_frecvente_SleepD) - 1)
print('P-value: ', p_value)

#concordance analysis for the Sleep.Disorder variable and a theoretical distribution
distributie_teoretica_SD = [0.1, 0.2, 0.7]

tabel_frecvente_SleepD2 = sleep_quality_check['Sleep.Disorder'].value_counts()

distributie_frecv_asteptate2 = [prob * len(sleep_quality_check)
                               for prob in distributie_teoretica_SD]

X_sq, p_value = chisquare(tabel_frecvente_SleepD2, distributie_frecv_asteptate2)

print('X-sq: ', X_sq)
print('df: ', len(tabel_frecvente_SleepD2) - 1)
print('P-value: ', p_value)


#4.1

#covariance analysis
analiza_cov = np.cov(data_sep_num, rowvar = False)

df_analiza_cov = pd.DataFrame(analiza_cov, columns = data_sep_num.columns,
                              index = data_sep_num.columns)

df_analiza_cov


#creating the correlation matrix (Pearson coefficient)

matrice_corelatieP = np.corrcoef(data_sep_num, rowvar = False)

df_matrice_corelatieP = pd.DataFrame(matrice_corelatieP, columns = data_sep_num.columns,
                                     index = data_sep_num.columns)

df_matrice_corelatieP

#Spearman coefficient matrix

matrice_corelatieS, _ = spearmanr(data_sep_num, axis = 0)

df_matrice_corelatieS = pd.DataFrame(matrice_corelatieS, columns = data_sep_num.columns,
                                     index = data_sep_num.columns)

df_matrice_corelatieS

#test of Pearson correlation

Stress_level = sleep_quality_check['Stress.Level']
Age = sleep_quality_check['Age']

coef_corelatie, p_value = pearsonr(Stress_level, Age)

print('Correlation coeficient r: ', coef_corelatie)
print('P-value: ', p_value)

#2

Stress_level = sleep_quality_check['Stress.Level']
Sleep_duration = sleep_quality_check['Sleep.Duration']

coef_corelatie, p_value = pearsonr(Stress_level, Sleep_duration)

print('Correlation coeficient r: ', coef_corelatie)
print('P-value: ', p_value)

#3

Stress_level = sleep_quality_check['Stress.Level']
QoS = sleep_quality_check['Quality.of.Sleep']

coef_corelatie, p_value = pearsonr(Stress_level, QoS)

print('Correlation coeficient r: ', coef_corelatie)
print('P-value: ', p_value)

#4

Stress_level = sleep_quality_check['Stress.Level']
PAL = sleep_quality_check['Physical.Activity.Level']

coef_corelatie, p_value = pearsonr(Stress_level, PAL)

print('Correlation coeficient r: ', coef_corelatie)
print('P-value: ', p_value)

#5

Stress_level = sleep_quality_check['Stress.Level']
Heart_rate = sleep_quality_check['Heart.Rate']

coef_corelatie, p_value = pearsonr(Stress_level, Heart_rate)

print('Correlation coeficient r: ', coef_corelatie)
print('P-value: ', p_value)

#4.2
#4.2.1 Simple linear regression model

x = sleep_quality_check['Quality.of.Sleep']
y = sleep_quality_check['Stress.Level']

x = sm.add_constant(x)
model = sm.OLS(y, x)

regresie_lin_simpla = model.fit()

print(regresie_lin_simpla.summary())

#Multiple regression model
import statsmodels.api as sm
x_multiple = sleep_quality_check[['Quality.of.Sleep', 'Age']]
y = sleep_quality_check['Stress.Level']

x_multiple = sm.add_constant(x)
model = sm.OLS(y, x_multiple)

regresie_lin_multipla = model.fit()

print(regresie_lin_multipla.summary())

#4.2.2 Non-linear regression model

Y = sleep_quality_check['Stress.Level']

sleep_quality_check['Quality.of.Sleep^2'] = sleep_quality_check['Quality.of.Sleep']**2

X_rn = sm.add_constant(sleep_quality_check[['Quality.of.Sleep', 'Quality.of.Sleep^2']])
model = sm.OLS(Y, X_rn)
regresie_neliniara = model.fit()

print(regresie_neliniara.summary())

#4.2.3

#Comparing 2 regression models

print(sm.stats.anova_lm(regresie_lin_simpla, regresie_lin_multipla))


#5 Mean estimation

print('Interval de incredere calculat cu functie integrata:',
      sm.DescrStatsW(sleep_quality_check['Stress.Level']).tconfint_mean())

#Mean estimation using a confidence interval
def interval_incredere_medie(data, incredere, val_testata = 0):
    a = np.array(data)
    n = len(a)
    media = np.mean(a)
    sem = scipy.stats.sem(a)
    h = sem * sp.stats.t.ppf((1 + incredere) / 2., n - 1)
    p_val = stats.ttest_1samp(a, val_testata)
    return media-h, media+h , p_val

    
interval_incredere = interval_incredere_medie(sleep_quality_check['Stress.Level'], 0.95)
print('Intervalul de incredere calculat prin intermediul functiei: ', interval_incredere)

#Mean test using a fixed value
t_calc, p_value = stats.ttest_1samp(sleep_quality_check['Stress.Level'], 5)

p_value_uni = p_value / 2 if t_calc > 0 else 1 - (p_value / 2)
mean = np.mean(sleep_quality_check['Stress.Level'])

print('T calculat: ', t_calc)
print('P_value: ', p_value)
print('Mean: ', mean)

#5.2.2 Testing the difference between two means (independent samples)
data_filtered_SQC = sleep_quality_check[sleep_quality_check['Sleep.Disorder'].isin([
    'Insomnia', 'Sleep Apnea'])]

insomnia_PC = data_filtered_SQC[data_filtered_SQC['Sleep.Disorder']
                                == 'Insomnia']['Stress.Level']
sleep_apnea_PC = data_filtered_SQC[data_filtered_SQC['Sleep.Disorder']
                                   == 'Sleep Apnea']['Stress.Level']

t_calc, p_value = stats.bartlett(insomnia_PC, sleep_apnea_PC)
#variance of the 2 groups
print('T-calc: ', t_calc)
print('P-value: ', p_value)

#welch two sample test stress.level pe grupe in functie de Sleep.Disorder

t_calc2, p_value2 = stats.ttest_ind(insomnia_PC, sleep_apnea_PC)
mean_IPC = np.mean(insomnia_PC)
mean_SAPC = np.mean(sleep_apnea_PC)

print('T-calc: ', t_calc2)
print('P-value: ', p_value2)
print('Mean of the category with Insomnia: ', mean_IPC)
print('Mean of the category with Sleep Apnea: ', mean_SAPC)

#testing 3 or more environments - ANOVA
sleep_quality_check.rename(columns={'Stress.Level': 'Stress_Level'}, inplace=True)
from statsmodels.formula.api import ols
model_means = ols('Stress_Level ~ sleep_disorder_stress', data=sleep_quality_check).fit()
import statsmodels.api as sm
anova_rezultat_test = sm.stats.anova_lm(model_means, typ=2)
print(anova_rezultat_test)
print(model_means.params)

#changing the column name in the dataset back
sleep_quality_check.rename(columns={'Stress_Level': 'Stress.Level'}, inplace=True)
print(sleep_quality_check.columns)
################################################


#hypothesis testing
#simple linear regression model testing - hypotheses
#1

#Heteroskedasticity tests - White test
from statsmodels.stats.diagnostic import het_white
white_test_rls = het_white(regresie_lin_simpla.resid, regresie_lin_simpla.model.exog)

labels = ['Test Statistic', 'Test Statistic P-value', 
         'F-Statistic', 'F-Test P-value']
print(dict(zip(labels, white_test_rls)))

#2
import statsmodels.stats.api as sm
test_BP=sm.het_breuschpagan(regresie_lin_simpla.resid, regresie_lin_simpla.model.exog)
labels = ['Test Statistic', 'Test Statistic P-value', 
         'F-Statistic', 'F-Test P-value']
print(dict(zip(labels, test_BP)))

#3
#autocorellation
#Breusch-Godfrey
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
bg_test = acorr_breusch_godfrey(regresie_lin_simpla)
labels = ['Test Statistic', 'Test Statistic P-value', 
         'F-Statistic', 'F-Test P-value']
print(dict(zip(labels, bg_test)))

#4 BP
import statsmodels.stats.api as sm
test_BP=sm.het_breuschpagan(regresie_lin_simpla.resid, regresie_lin_simpla.model.exog)
labels = ['Test Statistic', 'Test Statistic P-value', 
         'F-Statistic', 'F-Test P-value']
print(dict(zip(labels, test_BP)))

#RLM
#1

from statsmodels.stats.stattools import durbin_watson
dw_stat = durbin_watson(regresie_lin_multipla.resid)
print(f"Durbin-Watson Statistic: {dw_stat}")
#2
print(stats.jarque_bera(regresie_lin_multipla.resid))

#3
import statsmodels.stats.api as sm
test_BP_RLM = sm.het_breuschpagan(regresie_lin_multipla.resid, regresie_lin_multipla.model.exog)
labels = ['Test Statistic', 'Test Statistic P-value', 
         'F-Statistic', 'F-Test P-value']
print(dict(zip(labels, test_BP_RLM)))

#4
from statsmodels.stats.diagnostic import het_white
white_test_rlm = het_white(regresie_lin_multipla.resid, regresie_lin_multipla.model.exog)

labels = ['Test Statistic', 'Test Statistic P-value', 
         'F-Statistic', 'F-Test P-value']
print(dict(zip(labels, white_test_rlm)))

#R non-linear
#1
from statsmodels.stats.stattools import durbin_watson
dw_stat = durbin_watson(regresie_neliniara.resid)
print(f"Durbin-Watson Statistic: {dw_stat}")

#2
print(stats.jarque_bera(regresie_neliniara.resid))

#3
from statsmodels.stats.diagnostic import het_white
white_test_rn = het_white(regresie_neliniara.resid, regresie_neliniara.model.exog)

labels = ['Test Statistic', 'Test Statistic P-value', 
         'F-Statistic', 'F-Test P-value']
print(dict(zip(labels, white_test_rn)))
