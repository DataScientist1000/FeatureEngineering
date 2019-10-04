# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

# let's load the dataset with just a few columns and a few rows,
# to speed things up

use_cols = ['id', 'purpose', 'loan_status', 'home_ownership']

data = pd.read_csv(r"C:\Users\bberry\Documents\ReddyNotes\DataScientist1000\FeatureEngineering\Data\loan.csv", usecols=use_cols)\
.sample(10000, random_state=44)  # set a seed for reproducibility

data.head()
# let's inspect the variable home ownership,
# which indicates whether the borrowers own their home
# or if they are renting for example, among other things.

data.home_ownership.unique()

# let's make a bar plot, with the number of loans
# for each category of home ownership

fig = data['home_ownership'].value_counts().plot.bar()
fig.set_title('Home Ownership')
fig.set_ylabel('Number of customers')
"""
The majority of the borrowers either own their house on a mortgage or rent their property. A few borrowers own their home completely. The category 'Other' seems to be empty. To be completely sure, we could print the numbers as below.
"""
data['home_ownership'].value_counts()
"""
There are 2 borrowers that have other arrangements for their property. For example, they could live with their parents, or live in a hotel.
"""
# the "purpose" variable is another categorical variable
# that indicates how the borrowers intend to use the
# money they are borrowing, for example to improve their
# house, or to cancel previous debt.

data.purpose.unique()
# let's make a bar plot with the number of borrowers
# within each category

fig = data['purpose'].value_counts().plot.bar()
fig.set_title('Loan Purpose')
fig.set_ylabel('Number of customers')
"""
The majority of the borrowers intend to use the loan for 'debt consolidation' or to repay their 'credit cards'.
 This is quite a common among borrowers. What the borrowers intend to do is, to consolidate all the debt that they 
 have on different financial items, in one single debt, the new loan that they will take from Lending Club in this case. 
 This loan will usually provide an advantage to the borrower, either in the form of lower interest rates than a credit 
 card, for example, or longer repayment period.
"""
# let's look at one additional categorical variable,
# "loan status", which represents the current status
# of the loan. This is whether the loan is still active
# and being repaid, or if it was defaulted,
# or if it was fully paid among other things.

data.loan_status.unique()
# let's make a bar plot with the number of borrowers
# within each category

fig = data['loan_status'].value_counts().plot.bar()
fig.set_title('Status of the Loan')
fig.set_ylabel('Number of customers')

"""
We can see that the majority of the loans are active (current) and a big number have been 'Fully paid'. The remaining labels have the following meaning:
Late (16-30 days): customer missed a payment
Late (31-120 days): customer is behind in payments for more than a month
Charged off: the company declared that they will not be able to recover the money for that loan ( money is typically lost)
Issued: loan was granted but money not yet sent to borrower
In Grace Period: window of time agreed with customer to wait for payment, usually, when customer is behind in their payments
"""

# finally, let's look at a variable that is numerical,
# but its numbers have no real meaning, and therefore
# should be better considered as a categorical one.

data['id'].head()
"""
In this case, each id represents one customer. This number is assigned in order to identify the customer if needed, while maintaining confidentiality.
"""
# The variable has as many different id values as customers,
# in this case 10000, because we loaded randomly 
# 10000 rows/customers from the original dataset.

len(data['id'].unique())