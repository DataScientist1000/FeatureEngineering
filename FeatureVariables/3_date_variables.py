# -*- coding: utf-8 -*-
"""
Dates
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# let's load the Lending Club dataset with a few selected columns

use_cols = ['loan_amnt', 'grade', 'purpose', 'issue_d', 'last_pymnt_d']

data = pd.read_csv(r"C:\Users\bberry\Documents\ReddyNotes\DataScientist1000\FeatureEngineering\Data\loan.csv", usecols=use_cols)

data.head()
# let's inspect at the pandas type of object used to store the information

data.dtypes

# now let's parse the dates, currently coded as strings, into datetime format
# this will allow us to make some analysis afterwards

data['issue_dt'] = pd.to_datetime(data.issue_d)
data['last_pymnt_dt'] = pd.to_datetime(data.last_pymnt_d)

data[['issue_d', 'issue_dt', 'last_pymnt_d', 'last_pymnt_dt']].head()

"""
# let's see how much money Lending Club has disbursed
# (i.e., lent) over the years to the different risk
# markets (grade variable)
"""
fig = data.groupby(['issue_dt','grade'])['loan_amnt'].sum().unstack().plot(figsize = (14,8),linewidth = 2)
fig.set_title('Disbursed amount in time')
fig.set_ylabel('Disbursed Amount(US dollar)')

"""
Lending Club seems to have increased the amount of money lent from 2013 onwards. The tendency indicates that they continue to grow. In addition, we can see that their major business comes from lending money to C and B grades.
'A' grades are the lower risk borrowers, this is borrowers that most likely will be able to repay their loans, as they are typically in a better financial situation. Borrowers within this grade are typically charged lower interest rates.
E, F and G grades represent the riskier borrowers. Usually borrowers in somewhat tighter financial situations, or for whom there is not sufficient financial history to make a reliable credit assessment. They are typically charged higher rates, as the business, and therefore the investors, take a higher risk when lending them money.

Lending Club lends the biggest fraction to borrowers that intend to use that money to repay other debt or credit cards.
"""
