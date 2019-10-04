# -*- coding: utf-8 -*-
"""
Outlier
"""
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',None)

import seaborn as sns
import numpy as np

data = pd.read_csv(r"C:\Users\bberry\Documents\ReddyNotes\DataScientist1000\FeatureEngineering\Data\titanic_train.csv")
data.head()

sns.distplot(data.Age.fillna(95))
"""
There are 2 numerical variables in this dataset, Fare and Age. So let's go ahead and find out whether they present values that we could consider outliers.
Fare
"""
# First let's plot a histogram to get an idea of the distribution

fig = data.Fare.hist(bins=50)
fig.set_title('Fare Distribution')
fig.set_xlabel('Fare')
fig.set_ylabel('Number of Passengers')
"""
The distribution of Fare is skewed, so in principle, we shouldn't estimate outliers using the mean plus minus 3 standard deviations methods, which assumes a normal distribution of the data.
"""
# another way of visualising outliers is using boxplots and whiskers,
# which provides the quantiles (box) and inter-quantile range (whiskers),
# with the outliers sitting outside the error bars (whiskers).

# All the dots in the plot below are outliers according to the quantiles + 1.5 IQR rule

fig = data.boxplot(column='Fare')
fig.set_title('')
fig.set_xlabel('Survived')
fig.set_ylabel('Fare')

# let's look at the values of the quantiles so we can
# calculate the upper and lower boundaries for the outliers

# 25%, 50% and 75% in the output below indicate the
# 25th quantile, median and 75th quantile respectively

data.Fare.describe()

# Let's calculate the upper and lower boundaries
# to identify outliers according
# to interquantile proximity rule

IQR = data.Fare.quantile(0.75) - data.Fare.quantile(0.25)

Lower_fence = data.Fare.quantile(0.25) - (IQR * 1.5)
Upper_fence = data.Fare.quantile(0.75) + (IQR * 1.5)

Upper_fence, Lower_fence, IQR

# And if we are looking at really extreme values
# using the interquantile proximity rule

IQR = data.Fare.quantile(0.75) - data.Fare.quantile(0.25)

Lower_fence = data.Fare.quantile(0.25) - (IQR * 3)
Upper_fence = data.Fare.quantile(0.75) + (IQR * 3)

Upper_fence, Lower_fence, IQR
"""
The upper boundary for extreme outliers is a cost of 100 dollars for the Fare. The lower boundary is meaningless because there can't be a negative price for Fare.
"""
# lets look at the actual number of passengers on the upper Fare ranges

print('total passengers: {}'.format(data.shape[0]))

print('passengers that paid more than 65: {}'.format(
    data[data.Fare > 65].shape[0]))

print('passengers that paid more than 100: {}'.format(
    data[data.Fare > 100].shape[0]))

# and percentages of passengers
total_passengers = np.float(data.shape[0])

print('total passengers: {}'.format(data.shape[0] / total_passengers))

print('passengers that paid more than 65: {}'.format(
    data[data.Fare > 65].shape[0] / total_passengers))

print('passengers that paid more than 100: {}'.format(
    data[data.Fare > 100].shape[0] / total_passengers))

"""
When using the 3 times interquantile range itnerval to find outliers, we find that 6% of the passengers have paid extremely high fares. We can go ahead and investigate the nature of this outl
"""
# let's create a separate dataframe for high fare payers
high_fare_df = data[data.Fare>100]

# ticket: it indicates the people that bought their fares together
high_fare_df.groupby('Ticket')['Fare'].count()

"""
A group of people who bought their tickets together, say they were a family, would have the same ticket number. And the fare attached to them is no longer the individual Fare, rather the group Fare. This is why, we see this unusually high values:
    """
multiple_tickets = pd.concat(
    [
        high_fare_df.groupby('Ticket')['Fare'].count(),
        high_fare_df.groupby('Ticket')['Fare'].mean()
    ],
    axis=1)

multiple_tickets.columns = ['Ticket', 'Fare']
multiple_tickets.head(10)


"""
Therefore, the fare should be divided by the number of tickets bought together to find out the individual price. So we see how finding out and investigating the presence of outliers, can lead us to new insight about the dataset at hand. 
Go ahead and divide the Fare by the number of tickets bought together, and then repeat the finding outliers exercise on this newly created variable. Do you know how to do this in python?
If not, don't worry, I will show you how to calculate individual ticket price in the final lecture of this course in the section "Putting it all together".
For now, let's just go ahead and visualise a group of people that were seemingly travelling together and therefore bought the tickets together:
"""
    