# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 15:27:50 2018

@author: Nicholas Colonna
"I pledge my honor that I have abided by the Stevens Honor System." -Nicholas Colonna
"""
import pandas as pd
import pylab as pl
import numpy as np
import re
import os
import statsmodels.api as sm

#Modify the path:
path ="C:/Users/colon/Desktop/Fall 2018/QF 301 Advanced Time Series Analysis/Homework1"
os.chdir(path)

df = pd.read_csv("./data/credit-training.csv")


#1. Generate a table with number of null variables by variable
def print_null_freq(df):
    #for a given DataFrame, calculates how many values for each variable is null and prints the resulting table to stdout
    df_lng = pd.melt(df)
    null_variables = df_lng.value.isnull()
    return pd.crosstab(df_lng.variable, null_variables)

print (print_null_freq(df))

#2. Convert column names from camelCase into snake_case
def camel_to_snake(column_name):
    #converts a string that is camelCase into snake_case
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

print(df.columns)
df.columns = [camel_to_snake(col) for col in df.columns]
print(df.columns)

#3. After exploring the dataset, count the number of cases with all the following characteristics as a single query:
    # -people 35 or older
    # -who have no been delinquent in the past 2 years
    # -who have less than 10 open credit lines/loans
filter1 = (df.age >= 35) & (df.serious_dlqin2yrs == 0) & (df.number_of_open_credit_lines_and_loans < 10)
print()
print(len(df[filter1]))
    
#4. Repeat the exercise with all these restrictions as a single query:
    # -people who have been delinquent in the past 2 years
    # -are in the 90th percentile for monthly_income
filter2 = (df.serious_dlqin2yrs == 1) & (df.monthly_income >= df.monthly_income.quantile(.90))
print()
print(len(df[filter2]))

#5. Build and describe a forecasting model of the variable SeriousDlqin2yrs using logistic regression.
#regression on all variables
df = df.dropna()
y = df.serious_dlqin2yrs
x = df[['revolving_utilization_of_unsecured_lines','age','number_of_time30-59_days_past_due_not_worse','debt_ratio','monthly_income','number_of_open_credit_lines_and_loans','number_of_times90_days_late','number_real_estate_loans_or_lines','number_of_time60-89_days_past_due_not_worse','number_of_dependents']].copy()
logit_model = sm.Logit(y,x)
result = logit_model.fit()
print(result.summary2())

#revolving_utilization_of_unsecured_lines is not a significant variable, drop from regression
x = df[['age','number_of_time30-59_days_past_due_not_worse','debt_ratio','monthly_income','number_of_open_credit_lines_and_loans','number_of_times90_days_late','number_real_estate_loans_or_lines','number_of_time60-89_days_past_due_not_worse','number_of_dependents']].copy()
logit_model = sm.Logit(y,x)
result = logit_model.fit()
print(result.summary2())
#All variables in this regression are significant, therefore it can be a useful forecaster