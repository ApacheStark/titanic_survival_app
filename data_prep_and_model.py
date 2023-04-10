import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

import pickle

# Read in Kaggle Training Set
data = pd.read_csv('train.csv')

# Variables to include:
# Name input > Calc Len and Detect title
# Age
# Sex (if NB then F==0 and M==0)
# Family/friends bringing along 
# Which ticket would you like? (Group into OHE Pclass)

pclass_lut = {
        'A': 1,
        'B': 1,
        'C': 1,
        'D': 2,
        'E': 2,
        'F': 3,
        'G': 3,
        'Working On Ship': 3
    }

# Add together as family members
data["Fam_Size"] = data.SibSp + data.Parch

# Investgate possible correlations to Survival:
print(data.columns)
print()
print(pd.crosstab(data.Survived, data.Pclass))
print()
print(pd.crosstab(data.Survived, data.Sex))
print()
print(pd.crosstab(data.Survived, data.SibSp))
print()
print(pd.crosstab(data.Survived, data.Parch))


print()
print(pd.crosstab(data.Survived, data.Embarked))
print()

# Assuming first ticket is bunking ticket (naive):
data['Cabin_Group'] = [str(x)[0] for x in data['Cabin']]
print(pd.crosstab(data.Survived, data.Cabin_Group))
# remove T and reclass n as Working

print()
print(data.Name.head())

# Is simply the length of one's name important?
data['Name_len'] = [len(str(x).replace(' ', '')) for x in data.Name]
bx = data.boxplot(column='Name_len', by=['Sex', 'Survived'])
plt.show()

print(pd.crosstab(data.Cabin_Group, data.Pclass))


# Functions for feature creation:
# Do not simply create binary variable from either male or female as NB could be interpreted as F=0 and M=0 (gendered vars indepedent)
def if_fem(x):
    if x == 'female':
        return 1
    else:
        return 0

def if_mal(x):
    if x == 'male':
        return 1
    else:
        return 0 

# Subjective class casting, the overlap between pclass and cabin determines this:
def ticket_pclass1(x):
    if x in ['S', 'A','B']:
        return 1
    else:
        return 0

def ticket_pclass2(x):
    if x in ['C','D','E']:
        return 1
    else:
        return 0

def ticket_pclass3(x):
    if x in ['F','G', 'Working']:
        return 1
    else:
        return 0

# Investigate titles:
def get_title(x):
    if 'Dr.' in x or 'PhD' in x:
        return 'Dr'
    elif 'Lt.' in x or 'Major' in x or 'Col.' in x or 'Capt.' in x:
        return 'Vet'
    elif 'Rev' in x or 'Count' in x or 'Don.' in x or 'Count' in x or 'e.' in x or 'eer.' in x or 'Prince' in x or 'Princess' in x:
        return 'Royal'
    elif 'Mrs' in x:
        return 'Mrs'
    elif 'Ms' in x:
        return 'Ms'
    elif 'Miss' in x:
        return 'Ms'
    elif 'Mr' in x:
        return 'Mr' 
    elif 'Master' in x:
        return 'Master'
    else:
        return 'Unknown'

data['Title'] = data.Name.apply(get_title)
print(pd.crosstab(data.Survived, data.Title))

print(data[data.Title=='Unknown']['Name'])


keep_cols = [
    'Age', 'Fam_Size', 'Name_len'
]

sex_dummies = pd.get_dummies(data['Sex'])
title_dummies = pd.get_dummies(data['Title'])

data['Pclass'] = [str(x)+'_class' for x in data['Pclass']]

age_lut = data.groupby('Title').Age.mean().reset_index()
age_lut.columns = ['Title', 'Mean_Age']

def replace_age(x):
    if x['Age'] == x['Age']:
        return x['Age'] 
    else:
        return x['Mean_Age']

data = data.merge(age_lut)
data['Age'] = data.apply(replace_age, axis=1)

print(age_lut)


pclass_dummmies = pd.get_dummies(data['Pclass'])

print(data.shape)
X = pd.concat([data[keep_cols], sex_dummies, pclass_dummmies], axis=1)

print(X.shape)

print(X.info())



y = data['Survived']

# from sklearn.linear_model import LogisticRegression

clf = RandomForestClassifier(n_estimators=50,max_depth=None, min_samples_split=2,random_state=42)
# clf = LogisticRegression()

scores = cross_val_score(clf, X, y, cv=5)

clf.fit(X, y)

print(X.columns) 

name = 'Samuel Holt'
name_len = len(name)
ticket = 'G'
gender = 'male'

test_df = pd.DataFrame.from_dict(
    data = {
        'Age': [30],
        'Fam_Size': [1],
        'Name_len': [name_len],
        'gender': gender,
        'ticket': ticket
    }
)
    
test_df['female'] = test_df.gender.apply(if_fem)
test_df['male'] = test_df.gender.apply(if_mal)

test_df['1_class'] = test_df.ticket.apply(ticket_pclass1)
test_df['2_class'] = test_df.ticket.apply(ticket_pclass2)
test_df['3_class'] = test_df.ticket.apply(ticket_pclass3)

print(test_df)
model_cols = X.columns

# Store model columns for future reference:
print(model_cols)
pickle.dump(model_cols, open('model_columns.pkl', 'wb'))
# Ensure pickle succesful:
model_cols_in = pickle.load(open('model_columns.pkl', 'rb'))

# Test prediction on self:
print(model_cols_in)
chance = clf.predict_proba(test_df[X.columns])[0][1]
print(chance)
p_chance = round(chance * 100, 1)
print(f'{p_chance}% chance of survival')
print('Mean Score:', scores.mean())
# pickle.dump(clf, open('model.pkl', 'wb'))


# Follow up modelling approach:
# CV for optimisation
# XGBoost or catboost 
# Investigate further certain surnames and titles indicative of survival (much more fun for user interaction)