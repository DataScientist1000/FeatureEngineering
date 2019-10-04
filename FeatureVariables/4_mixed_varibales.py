# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel(r"C:\Users\bberry\Documents\ReddyNotes\DataScientist1000\FeatureEngineering\Data\mixed_data.xlsx")

data.head()
data.shape

# 'A': couldn't identify the person
# 'B': no relevant data
# 'C': person seems not to have any account open

data.open_il_24m.unique()

# Now, let's make a bar plot showing the different number of 
# borrowers for each of the values of the mixed variable

fig = data.open_il_24m.value_counts().plot.bar()
fig.set_title('Number of installment accounts open')
fig.set_ylabel('Number of borrowers')