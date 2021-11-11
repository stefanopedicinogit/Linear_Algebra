#!/usr/bin/env python
# coding: utf-8

# # Statement

# The Sure Tomorrow insurance company wants to solve several tasks with the help of Machine Learning and you are asked to evaluate that possibility.

# # Data Preprocessing & Exploration
# 
# ## Initialization

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import Counter
from scipy.spatial import distance
import math

import seaborn as sns

import sklearn.linear_model
import sklearn.metrics
import sklearn.neighbors
import sklearn.preprocessing
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

from sklearn.model_selection import train_test_split

from IPython.display import display


# ## Load Data

# Load data and conduct a basic check that it's free from obvious issues.

# In[2]:


df = pd.read_csv('/datasets/insurance_us.csv')


# We rename the colums to make the code look more consistent with its style.

# In[3]:


df = df.rename(columns={'Gender': 'gender', 'Age': 'age', 'Salary': 'income', 'Family members': 'family_members', 'Insurance benefits': 'insurance_benefits'})


# In[4]:


display(df.sample(10, random_state=0))


# In[5]:


df.info()


# In[6]:


# we may want to fix the age type (from float to int) though this is not critical
# write your conversion here if you choose:
df.age = df.age.astype('int16')


# In[7]:


# check to see that the conversion was successful
df.info()


# In[8]:


# now have a look at the data's descriptive statistics. 
# Does everything look okay?


# In[9]:


display(df.describe())


# In[10]:


print('NaNs:', df.isna().sum().sum())


# ## EDA

# Let's quickly check whether there are certain groups of customers by looking at the pair plot.

# In[11]:


g = sns.pairplot(df)
g.fig.set_size_inches(12, 12)


# Ok, it is a bit difficult to spot obvious groups (clusters) as it is difficult to combine several variables simultaneously (to analyze multivariate distributions). That's where LA and ML can be quite handy.

# In[12]:


features_plot= df.drop('insurance_benefits', axis=1)
target_plot= df['insurance_benefits']
features_plot = np.array(features_plot)
target_plot = np.array(target_plot)


# In[13]:


#Understanding insurance benefits classes distribution.
cmap = ListedColormap(['red', 'orange',  'yellow', 'green', 'blue','magenta'])
plt.figure(figsize=(16,5))
plt.scatter(features_plot[:, 1], features_plot[:, 2], c=target_plot, cmap=cmap, edgecolor='k',s=20)
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()


# Scatter plot of insurance_benefits classes distribution related to Income and Age columns.

# In[14]:


#Graph proof. Identifying the single point in magenta (scatterplot above) in the dataset.
print(df.query('insurance_benefits == 5'))


# # Task 1. Similar Customers

# In[15]:


feature_names = ['gender', 'age', 'income', 'family_members']


# In[16]:


def get_knn(df, n, k, metric):
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k, radius=0.4, metric=metric)
    nbrs.fit(df[feature_names],df['insurance_benefits'])
    nbrs_distances, nbrs_indices = nbrs.kneighbors([df.iloc[n][feature_names]], k, return_distance=True)
    
    df_res = pd.concat([
        df.iloc[nbrs_indices[0]], 
        pd.DataFrame(nbrs_distances.T, index=nbrs_indices[0], columns=['distance'])
        ], axis=1)
    
    return df_res


# In[17]:


#Testing Euclidean distance on unscaled data
display(get_knn(df, 398, 5, distance.euclidean))


# In[18]:


#Testing Manhattan distance on unscaled data
display(get_knn(df, 398, 5, distance.cityblock))


# Scaling the data.

# In[19]:


#Scaling the data
feature_names = ['gender', 'age', 'income', 'family_members']

transformer_mas = sklearn.preprocessing.MaxAbsScaler().fit(df[feature_names].to_numpy())

df_scaled = df.copy()
df_scaled.loc[:, feature_names] = transformer_mas.transform(df[feature_names].to_numpy())


# In[20]:


display(df_scaled.sample(5, random_state=0))


# Now, let's get similar records for a given one for every combination

# In[21]:


#Testing Euclidean distance on unscaled data
display(get_knn(df_scaled, 398, 5, distance.euclidean))


# In[22]:


#Testing Manhattan distance on unscaled data
display(get_knn(df_scaled, 398, 5, distance.cityblock))


# **Does the data being not scaled affect the kNN algorithm? If so, how does that appear?** 
# 
# The two Dataframes, scaled and unscaled returned different records within the datasets.\
# - The unscaled one, for ID 398 returned as nearest points 2599, 152, 2140, 2039.
# - The scaled data, for ID 398 returned as nearest points 4366, 3242, 2041, 2934.
# 
# The difference on the calculation is that in:
# - On Scaled data, the higher values have an higher weight so the nearest points are choosen taking in consideration the column weight itself.
# - On the Unscaled data, every column has the same weight, we calculations take in consideration the entirely DataFrame.
# 
# The results returned by the two Euclidean and Manhattan distance are different among them regarding the data scaling. They returned two different result on scaled and unscaled data. They look even different taking the values in columns.
# 
# For example, while **'family_members'** column:
# - on **Unscaled** data had **all different values** [2,3,1,3] 
# - the **Scaled** one, has **the same values** represented in the scaled numbers as [0.166667]. 
# 
# The contrary happend on **'income'** column:
# - while on **Unscaled** data we have **all equal values** [37700.0] 
# - on the **Scaled** dataset it returned **all different scaled values** [0.482278, 0.463291, 0.478481, 0.458228].
# 
# In my opinion, this happened because on Unscaled data, an high value like 377000 has a major weight compared to the other column values while calculating distances. 
# - In the first example, turned out that all the values with the same income of the choosen data point were closer then the others because the income value is heaviest then the others.
# - In the second example instead, the weights are equally distribuited for this reason the calculations of the closest point is done on every columns.

# **How similar are the results using the Manhattan distance metric (regardless of the scaling)?** 
# 
# For the data **Unscaled** the two distances returned the same data point. While the Euclidean distance is in floating numbers instead the Manhattan one, presents only integers. 
# 
# The point 2599 is just rounded to the next integer(2.236068 to 3), while the others change a little bit, the 152 from 4.472136 become 6.0 and the 2039 and 2140 from 4.582576 and 6.082763 become both 7.
# 
# For the **Scaled** data the points are the same as well, except for the last. While Euclidean returned as last nearest point the number 2238 the Manhattan returned the id 2934.
# 
# Here we can see that the distances are equals if we consider the number 4366 and 3242 while they differs on the two other data points but only really slightly.

# # Task 2. Is Customer Likely to Receive Insurance Benefit?

# In[23]:


# Calculate the target
df['insurance_benefits_received'] = df['insurance_benefits'] >= 1
df.loc[df['insurance_benefits_received'] == True, 'insurance_benefits_received'] = 1
df.loc[df['insurance_benefits_received'] == False, 'insurance_benefits_received'] = 0
df_scaled['insurance_benefits_received'] = df_scaled['insurance_benefits'] >= 1
df_scaled.loc[df['insurance_benefits_received'] == True, 'insurance_benefits_received'] = 1
df_scaled.loc[df['insurance_benefits_received'] == False, 'insurance_benefits_received'] = 0


# In[24]:


# check for the class imbalance with value_counts()
print(df['insurance_benefits_received'].value_counts())
print()
print(df['insurance_benefits_received'].value_counts(normalize=True))


# In[25]:


#Assigning features and target Unscaled data
features = df.drop(['insurance_benefits_received', 'insurance_benefits'], axis=1)
target = df['insurance_benefits_received']
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.3, random_state=0)
print(features_train.shape, target_train.shape)
print(features_test.shape, target_test.shape)


# In[26]:


#Scaling data.
scaler = MaxAbsScaler()
scaler.fit(features_train)
features_train_scaled = scaler.transform(features_train)
features_test_scaled = scaler.transform(features_test)


# In[27]:


def eval_classifier(target_test, predictions):
    
    f1_score = sklearn.metrics.f1_score(target_test, predictions)
    print(f'F1: {f1_score:.2f}')
    
    cm = sklearn.metrics.confusion_matrix(target_test, predictions)
    print('Confusion Matrix')
    print(cm)


# In[28]:


# generating output of a random model

def rnd_model_predict(P, size, seed=42):

    rng = np.random.default_rng(seed=seed)
    return rng.binomial(n=1, p=P, size=size)


# $$
# P\{\text{insurance benefit received}\}=\frac{\text{number of clients received any insurance benefit}}{\text{total number of clients}}.
# $$
# 

# In[29]:


#Confusion matrix of dummy rnd_model predictions. Unscaled data.
for P in [0, df['insurance_benefits_received'].sum() / len(df), 0.5, 1]:
    print(f'The probability: {P:.2f}')
    y_pred_rnd =  rnd_model_predict(P, features.shape[0])
        
    eval_classifier(df['insurance_benefits_received'], y_pred_rnd)
    
    print()


# In[30]:


#Confusion matrix of dummy rnd_model predictions. Scaled data
for P in [0, df_scaled['insurance_benefits_received'].sum() / len(df_scaled), 0.5, 1]:
    print(f'The probability: {P:.2f}')
    y_pred_rnd =  rnd_model_predict(P, df_scaled.shape[0])
    
    eval_classifier(df_scaled['insurance_benefits_received'], y_pred_rnd)
    
    print()


# In[31]:


#K-NeighborsClassifier for k in range 10 + f1_score. Unscaled data
print('Unscaled data.')
unscaled_f1 = []
for k in range(1,11):
    knc = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
    knc.fit(features_train, target_train)
    knc_pred = knc.predict(features_test)
    f1_score = sklearn.metrics.f1_score(target_test, knc_pred)
    unscaled_f1.append(f1_score)
    print(f'N_neighbors: {k}, F1: {f1_score:.2f}')


# In[32]:


#K-NeighborsClassifier for k in range 10 + f1_score. Scaled data
print('Scaled data.')
scaled_f1 = []
for k in range(1,11):
    knc = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
    knc.fit(features_train_scaled, target_train)
    knc_pred = knc.predict(features_test_scaled)
    f1_score = sklearn.metrics.f1_score(target_test, knc_pred)
    scaled_f1.append(f1_score)
    print(f'N_neighbors: {k}, F1: {f1_score:.2f}')


# In[34]:


# Plot the results 
plt.figure(figsize=(12,6))
plt.plot(scaled_f1, label = "Unscaled F1")
plt.plot(unscaled_f1, label = 'Scaled F1')
plt.xlabel('# of Nearest Neighbors (k)')
plt.ylabel('F1 Score (%)')
plt.legend()
plt.show()


# # Task 3. Regression (with Linear Regression)

# In[35]:


#Creating a class of LinearRegression.
class MyLinearRegression:
    
    def __init__(self):
        self.weights = None
    
    def fit(self, X, y):
        # adding the unities
        X2 = np.append(np.ones([len(X), 1]), X, axis=1)
        weights = np.linalg.inv(X2.T.dot(X2)).dot(X2.T).dot(y)
        self.weights = weights[1:]
        self.bias = weights[0]

    def predict(self, X):
        # adding the unities
        X2 = np.append(np.ones([len(X), 1]), X, axis=1)
        y_pred = X.dot(self.weights) + self.bias
        return y_pred


# In[36]:


#def function to evaluate the classifier.
def eval_regressor(y_true, y_pred):
    
    rmse = math.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
    print(f'RMSE: {rmse:.2f}')
    
    r2_score = math.sqrt(sklearn.metrics.r2_score(y_true, y_pred))
    print(f'R2: {r2_score:.2f}')    


# In[37]:


#Testing the LR on the Unscaled data
X = df[['age', 'gender', 'income', 'family_members']].to_numpy()
y = df['insurance_benefits'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

lr = MyLinearRegression()

lr.fit(X_train, y_train)
print(lr.weights)

y_test_pred = lr.predict(X_test)
eval_regressor(y_test, y_test_pred)


# In[38]:


#Testing LR on Scaled data.
X_scaled = df_scaled[['age', 'gender', 'income', 'family_members']].to_numpy()
y_scaled = df_scaled['insurance_benefits'].to_numpy()

X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=12345)

lr = MyLinearRegression()

lr.fit(X_train_scaled, y_train_scaled)
print(lr.weights)

y_test_pred_scaled = lr.predict(X_test_scaled)
eval_regressor(y_test_scaled, y_test_pred_scaled)


# # Task 4. Obfuscating Data

# In[39]:


#Creating a database taking in consideration only the 4 features columns.
personal_info_column_list = ['gender', 'age', 'income', 'family_members']
df_pn = df[personal_info_column_list]


# In[40]:


#Generating a matrix X 
X = df_pn.to_numpy()


# Generating a random matrix $P$.

# In[41]:


#Generating a matrix P 
rng = np.random.default_rng(seed=42)
P = rng.random(size=(X.shape[1], X.shape[1]))
print(P)


# Checking the matrix $P$ is invertible

# In[42]:


#Checking the invertibility of P
print(np.linalg.inv(P))


# Can you guess the customers' ages or income after the transformation?

# In[43]:


#Transforming the entire df in X1 (obfuscating data)
X1=X[:,:].dot(P)
print(X1)


#     No I can't anymore distinguish and recognize numbers inside the dataframe. 

# Can you recover the original data from $X'$ if you know $P$? Try to check that with calculations by moving $P$ from the right side of the formula above to the left one. The rules of matrix multiplcation are really helpful here.

# In[44]:


#Reversing from X1 the matrice X
X = X1.dot(np.linalg.inv(P))
print(X)


#     Multiplying the obfuscated matrix for the inverse of P we are able to reobtain the X matrice. 

# Print all three cases for a few customers
# - The original data
# - The transformed one
# - The reversed (recovered) one

# In[45]:


#Dropping unecessary columns for the matrice.
df_obfuscation = df_pn[:4]
df_obfuscation = df_pn
display(df_obfuscation)


# In[46]:


#Original data
raw = np.array(df_obfuscation)
print(raw)


# In[47]:


#Obfuscated data
obf = raw.dot(P)
print(obf)


# In[48]:


#Recovered
rec = obf.dot(np.linalg.inv(P))
print(rec)


# You can probably see that some values are not exactly the same as they are in the original data. What might be the reason for that?

# From the results received I obtained really similar values to the one I had in the beginning. They change in infinitesimal floating points, this depends by the binary computation of computers. But basically 1.67952800e-12 mean a value near to the zero.

# ## Proof That Data Obfuscation Can Work with LR

# **Analytical proof**

# $w_P = [(XP)^T XP]^{-1} (XP)^T y$ \
# $w_P = [P^TX^T XP]^{-1} P^TX^T y$ \
# $w_P = P^{-1}X^{-1}X^{T-1}P^{T-1} P^TX^T y$ \
# $P^{T-1}P^T = I$ \

# $w_P = P^{-1}(X^TX)^{-1}X^Ty$\
# $w = (X^TX)^{-1}X^Ty$

# $w$ and $wP$ are linked by $P^{-1}$ that can be write as: $w_p = P^{-1}w$.
# 
# Analitically the predictions formula can be written as:\
# $y = Xw$
# 
# Instead the prediction of $w_p$ are:\
# $y_p = X_pw_p = XPP^{-1}w$ = $Xw$ 

# - Since the formula for $y$ and $y_p$ is the same, we can say that the original predictions are the same of the obfuscate dataset ones.

# <table>
# <tr>
# <td>Distributivity</td><td>$A(B+C)=AB+AC$</td>
# </tr>
# <tr>
# <td>Non-commutativity</td><td>$AB \neq BA$</td>
# </tr>
# <tr>
# <td>Associative property of multiplication</td><td>$(AB)C = A(BC)$</td>
# </tr>
# <tr>
# <td>Multiplicative identity property</td><td>$IA = AI = A$</td>
# </tr>
# <tr>
# <td></td><td>$A^{-1}A = AA^{-1} = I$
# </td>
# </tr>    
# <tr>
# <td></td><td>$(AB)^{-1} = B^{-1}A^{-1}$</td>
# </tr>    
# <tr>
# <td>Reversivity of the transpose of a product of matrices,</td><td>$(AB)^T = B^TA^T$</td>
# </tr>    
# </table>

# ## Test Linear Regression With Data Obfuscation

# In[49]:


class MyLinearRegression_obf:
    
    def __init__(self, data):
        self.weights = None
        self.data = data
        
    def fit(self, X, y):
        # adding the unities
        X2 = np.append(np.ones([len(X), 1]), X, axis=1)
        weights = np.linalg.inv(X2.T.dot(X2)).dot(X2.T).dot(y)
        self.weights = weights[1:]
        self.bias = weights[0]

    def predict(self, X):
        # adding the unities
        X2 = np.append(np.ones([len(X), 1]), X, axis=1)
        y_pred = X.dot(self.weights) + self.bias
        return y_pred


# In[50]:


#MyLinearRegression_obf on the normal dataset.
X = X
y = df['insurance_benefits'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

lr = MyLinearRegression_obf(X)

lr.fit(X_train, y_train)
print(lr.weights)

y_test_pred = lr.predict(X_test)
eval_regressor(y_test, y_test_pred)


# In[51]:


#MyLinearRegression_obf on the obfuscated dataset.
X = X1
y = df['insurance_benefits'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

lr = MyLinearRegression_obf(X1)

lr.fit(X_train, y_train)
print(lr.weights)

y_test_pred = lr.predict(X_test)
eval_regressor(y_test, y_test_pred)


# # Conclusions

# I were able thanks to NearestNeighbors imported library to obtain the K neighbors for a given dataframe point, we can state doing this that the scaling of our data changes totally our results. When the data are not scaled at all, the results we got are similar on the bigger values ('Income' in our case) cause the bigger values have an higher weights on the results. While scaling data we can return the right neighbors taking in consideration all the features.

# I indentified the target of this task being 1 for any person that received at least 1 insurance benefit, and 0 for people who doesn't. The distribution of those values in our dataset is:  88.72% (0) and 11.28% (1).
# The results of our dummy model were in general really low values. 
# Using KNeiborghsClassifier I plotted the f1 score for any values of K, on Unscaled data I obtained the higher value of 60% with k==1, while scaling the data, our predictions resulted perfect obtaining a value of 100%, this mean that the precision and the recall of the model itself are the best possible percentages.

# I built an own LinearRegression model evaluating it on two different metrics, first Root mean squared error (RMSE) and R2 score, performing predictions on scaled and unscaled data, doesn't make any difference and the **RMSE** was calculated in **0.34** that is modest result, it mean that on a scale from 0 to 1 the mistakes made from the model predictions are 34%. **R2** score in a range of 100, returned **66%**.

# In this task was required to obfusc the data in our hands, to make that I created a matrice P wich was invertible and multiplied it for our features matrice X. The dot product returned another matrice X1, that hides data letting them being not recognizable. Is possible to revert the process to re-obtain the starting values. I proved that our LR model performs the predictions in the same way on the obfuscated data having the same quality and the same metrics values.
