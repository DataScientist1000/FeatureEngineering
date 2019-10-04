# -*- coding: utf-8 -*-
"""
Missing values
In python, the missing values are stored as NaN, see for example the first row for the variable Cabin.
"""
import numpy as np 
import pandas as pd

pd.set_option('display.max_columns',None)
data = pd.read_csv(r"C:\Users\bberry\Documents\ReddyNotes\DataScientist1000\FeatureEngineering\Data\titanic_train.csv")

# you can determine the total number of missing values using
# the isnull method plus the sum method on the dataframe

data.isnull().sum()
# alternatively, you can call the mean method after isnull
# to visualise the percentage of the dataset that 
# contains missing values for each variable

data.isnull().mean()
"""
We can see that there are missing data in the variables Age, Cabin (in which the passenger was travelling) and Embarked, which is the port from which the passenger got into the Titanic.
"""
"""
Missing data Not At Random (MNAR): Systematic missing values
In this dataset, both the missing values of the variables Cabin and Age, were introduced systematically. For many of the people who did not survive, the age they had or the cabin they were staying in, could not be established. The people who survived could be asked for that information.
Can we infer this by looking at the data?
In a situation like this, we could expect a greater number of missing values for people who did not survive.
Let's have a look.
"""
# we create a dummy variable that indicates whether the value
# of the variable cabin is missing

data['cabin_null'] = np.where(data.Cabin.isnull(), 1, 0)

# find percentage of null values
data.cabin_null.mean()
#As expected, this value coincides with the one observed above when we called the .isnull().mean() method on the dataset.
# and then we evaluate the mean of the missing values in
# cabin for the people who survived vs the non-survivors.

# group data by Survived vs Non-Survived
# and find nulls for cabin
data.groupby(['Survived'])['cabin_null'].mean()
"""
We observe that the percentage of missing values is higher for people who did not survive (0.87), respect to people that survived (0.60).
This finding is aligned with our hypothesis that the data is missing because after the people died, the information could not be retrieved.

Having said this, to truly underpin whether the data is missing not at random, we would need to get extremely familiar with the way data was collected. Analysing datasets, can only point us in the right direction or help us build assumptions.
"""
# we repeat the exercise for the variable age:
# First we create a dummy variable that indicates
# whether the value of the variable Age is missing

data['age_null'] = np.where(data.Age.isnull(), 1, 0)

# and then look at the mean in the different survival groups:
# there are more NaN for the people who did not survive
data.groupby(['Survived'])['age_null'].mean()

"""
Again, we observe an increase in missing data for the people who did not survive the tragedy. The analysis therefore suggests: 
There is a systematic loss of data: people who did not survive tend to have more information missing. Presumably, the method chosen to gather the information, contributes to the generation of these missing data.

Missing data Completely At Random (MCAR)
In the titanic dataset, there were also missing values for the variable Embarked, let's have a look.
"""
# slice the dataframe to show only those observations
# with missing value for Embarked

data[data.Embarked.isnull()]
"""
These 2 women were travelling together, Miss Icard was the maid of Mrs Stone.
    
A priori, there does not seem to be an indication that the missing information in the variable Embarked is depending on any other variable, and the fact that these women survived, means that they could have been asked for this information.

Very likely this missingness was generated at the time of building the dataset and therefore we could assume that it is completely random. We can assume that the probability of data being missing for these 2 women is the same as the probability for this variable to be missing for any other person. Of course this will be hard, if possible at all, to prove.
Missing data At Random (MAR)
For this example, I will use the Lending Club loan book. I will look specifically at the variables employer name (emp_title) and years in employment (emp_length), declared by the borrowers at the time of applying for a loan. The former refers to the name of the company for which the borrower works, the second one to how many years the borrower has worked for named company.
Here I will show an example, in which a data point missing in one variable (emp_title) depends on the value entered on the other variable (emp_lenght).
"""


# let's load the columns of interest from the Lending Club loan book dataset

data_loan=pd.read_csv(r"C:\Users\bberry\Documents\ReddyNotes\DataScientist1000\FeatureEngineering\Data\loan.csv", usecols=['emp_title','emp_length'])
data_loan.head()


# let's check the amount of missing data
data_loan.isnull().mean()

# let's peek at the different employer names

print('Number of different employer names: {}'.format(len(data_loan.emp_title.unique())))
data_loan.emp_title.unique()[0:20]

# let's inspect the variable emp_length
data_loan.emp_length.unique()

# let's look at the percentage of borrowers within
# each label / category of the emp_length variable

data_loan.emp_length.value_counts() / len(data_loan)

# the variable emp_length has many categories. I will summarise it
# into 3 for simplicity:'0-10 years' or '10+ years' or 'n/a'

# let's build a dictionary to re-map emp_length to just 3 categories:

length_dict = {k:'0-10 years' for k in data_loan.emp_length.unique()}
length_dict['10+ years']='10+ years'
length_dict['n/a']='n/a'

# let's look at the dictionary
length_dict


data_loan['emp_length_redefined'] = data_loan.emp_length.map(length_dict)
data_loan.emp_length_redefined.unique()

# let's calculate the proportion of working years
# with same employer for those who miss data on employer name

# number of borrowers for whom employer name is missing
value = len(data_loan[data_loan.emp_title.isnull()])

# % of borrowers for whom employer name is missing 
# within each category of employment length
data_loan[data_loan.emp_title.isnull()].groupby(['emp_length_redefined'])['emp_length'].count().sort_values() / value


# let's do the same for those bororwers who reported
# the employer name

# number of borrowers for whom employer name is present
value = len(data_loan.dropna(subset=['emp_title']))

# % of borrowers within each category
data_loan.dropna(subset=['emp_title']).groupby(['emp_length_redefined'])['emp_length'].count().sort_values() / value

"""
The number of borrowers who have reported an employer name and indicate n/a as employment length are minimal. Further supporting the idea that the missing values in employment length and employment length are related.

'n/a' in 'employment length' could be supplied by people who are retired, or students, or self-employed. In all of those cases there would not be a number of years at employment to provide, therefore the customer would enter 'n/a' and leave empty the form at the side of 'employer_name'.

In a scenario like this, a missing value in the variable emp_title depends on or is related to the 'n/a' label in the variable emp_length. And, this missing value nature is, in principle, independent of the variable we want to predict (in this case whether the borrower will repay their loan). How this will affect the predictions is unknown.
"""










































