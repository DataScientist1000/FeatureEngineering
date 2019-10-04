# -*- coding: utf-8 -*-

"""
Numerical Variables 
 > Discrete
 > Continuous
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load the dataset 
ds = pd.read_csv(r"C:\Users\bberry\Documents\ReddyNotes\DataScientist1000\FeatureEngineering\Data\titanic_train.csv")
numeric_columns = ds[['Survived','Pclass','SibSp','Parch','Fare']]

numeric_columns.head()
numeric_columns.Pclass.unique()
numeric_columns.Fare.unique()
numeric_columns.sort_values()
#------------------------------------------------------------
# load the dataset 
loanData = pd.read_csv(r"C:\Users\bberry\Documents\ReddyNotes\DataScientist1000\FeatureEngineering\Data\sampleLoandata.csv")
use_cols = ['loan_amnt', 'int_rate', 'annual_inc', 'open_acc', 'loan_status','open_il_12m']
data = pd.read_csv(r"C:\Users\bberry\Documents\ReddyNotes\DataScientist1000\FeatureEngineering\Data\loan.csv", usecols=use_cols).sample(
        50000, random_state=44)  # set a seed for reproducibility
data.head()
#---Continuous Variables---
data.loan_amnt.unique()

# let's make an histogram to get familiar with the
# distribution of the variable
fig = data.loan_amnt.hist(bins=50)
fig.set_title('Loan Amount Requested')
fig.set_xlabel('Loan Amount')
fig.set_ylabel('Number of Loans')

"""
The values of the variable vary across the entire range of the variable. This is characteristic of continuous variables.
The taller bars correspond to loan sizes of 10000, 15000, 20000, and 35000. There are more loans disbursed for those loan amount values. This indicates that most people tend to ask for these loan amounts. Likely, these particular loan amounts are pre-determined and offered as such in the Lending Club website.
Less frequent loan values, like 23,000 or 33,000 could be requested by people who require a specific amount of money for a definite purpose.
"""
#-- checking the int_rate
data.int_rate.unique()
print('The # of unique intrest rates : ', len(data.int_rate.unique()))
      
fig = data.int_rate.hist(bins = 30)
fig.set_title('Intrest Rate ')
fig.set_xlabel('IR')
fig.set_ylabel('Number of loans')      
#Again, we see that the values of the variable vary continuously across the variable range.

# and now,let's explore the income declared by the customers,
# that is, how much they earn yearly.
fig = data.annual_inc.hist(bins=100)
fig.set_xlim(0, 400000)
fig.set_title("Customer's Annual Income")
fig.set_xlabel('Annual Income')
fig.set_ylabel('Number of Customers')
"""
The majority of salaries are concentrated towards values in the range 30-70 k, with only a few customers earning higher salaries. Again, the values of the variable, vary continuosly across the variable range.
Discrete Variables
Let's explore the variable "Number of open credit lines in the borrower's credit file" (open_acc in the dataset). This is, the total number of credit items (for example, credit cards, car loans, mortgages, etc) that is known for that borrower. By definition it is a discrete variable, because a borrower can have 1 credit card, but not 3.5 credit cards.
"""
# let's make an histogram to get familiar with the
# distribution of the variable

fig = data.open_acc.hist(bins=100)
fig.set_xlim(0, 30)
fig.set_title('Number of open accounts')
fig.set_xlabel('Number of open accounts')
fig.set_ylabel('Number of Customers')

# let's inspect the variable values
data.open_il_12m.unique()
# let's make an histogram to get familiar with the
# distribution of the variable

fig = data.open_il_12m.hist(bins=50)
fig.set_title('Number of installment accounts opened in past 12 months')
fig.set_xlabel('Number of installment accounts opened in past 12 months')
fig.set_ylabel('Number of Borrowers')
#--The majority of the borrowers have none or 1 installment account, with only a few borrowers having more than 2.
"""
A variation of discrete variables: the binary variable
Binary variables, are discrete variables, that can take only 2 values, therefore binary.
In the next cells I will create an additional variable, called defaulted, to capture the number of loans that have defaulted. A defaulted loan is a loan that a customer has failed to re-pay and the money is lost.
The variable takes the values 0 where the loans are ok and being re-paid regularly, or 1, when the borrower has confirmed that will not be able to re-pay the borrowed amount.
"""
# let's inspect the values of the variable loan status
data.loan_status.unique()
# let's create one additional variable called defaulted.
# This variable indicates if the loan has defaulted, which means,
# if the borrower failed to re-pay the loan, and the money
# is deemed lost.

data['defaulted'] = np.where(data.loan_status.isin(['Default']), 1, 0)
data.defaulted.mean()
# the new variable takes the value of 0
# if the loan is not defaulted

data.head()
# the new variable takes the value 1 for loans that
# are defaulted

data[data.loan_status.isin(['Default'])].head()
# A binary variable, can take 2 values. For example,
# the variable defaulted that we just created:
# either the loan is defaulted (1) or not (0)

data.defaulted.unique()
# let's make a histogram, although histograms for
# binary variables do not make a lot of sense

fig = data.defaulted.hist()
fig.set_xlim(0, 2)
fig.set_title('Defaulted accounts')
fig.set_xlabel('Defaulted')
fig.set_ylabel('Number of Loans')
#As we can see, the variable shows only 2 values, 0 and 1, and the majority of the loans are ok.