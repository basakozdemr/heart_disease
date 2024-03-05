#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[37]:


df = pd.read_csv("heart_2020_cleaned.csv")


# In[38]:


df


# In[19]:


df.info()


# In[20]:


df.shape


# In[7]:


df['AgeCategory'] = df['AgeCategory'].replace(['18-24', '25-29', '30-34', '35-39', '40-44', '25-29', '30-34', '35-40', '41-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
df['Race'] = df['Race'].replace(['White', 'Hispanic', 'Black', 'Asian', 'American Indian/Alaskan Native', 'Other'], [0, 1, 2, 3, 4, 5])
df['Diabetic'] = df['Diabetic'].replace(['No', 'Yes (during pregnancy)', 'No, borderline diabetes', 'Yes'], [0, 1, 2, 3])
df['GenHealth'] = df['GenHealth'].replace(['Excellent', 'Very good', 'Good', 'Fair','Poor'], [0, 1, 2, 3, 4])
df['PhysicalActivity'] = df['PhysicalActivity'].replace(['No', 'Yes'], [0, 1]) # do you play any sports?
df['Asthma'] = df['Asthma'].replace(['No', 'Yes'], [0, 1])
df['KidneyDisease'] = df['KidneyDisease'].replace(['No', 'Yes'], [0, 1])
df['SkinCancer'] = df['SkinCancer'].replace(['No', 'Yes'], [0, 1])
df['DiffWalking'] = df['DiffWalking'].replace(['No', 'Yes'], [0, 1])
df['Sex'] = df['Sex'].replace(['Female', 'Male'], [0, 1])
df['HeartDisease'] = df['HeartDisease'].replace(['No', 'Yes'], [0, 1])
df['Smoking'] = df['Smoking'].replace(['No', 'Yes'], [0, 1])
df['AlcoholDrinking'] = df['AlcoholDrinking'].replace(['No', 'Yes'], [0, 1])
df['Stroke'] = df['Stroke'].replace(['No', 'Yes'], [0, 1])


# In[8]:


df.head()


# In[10]:


df.info()


# In[11]:


x =  df[["BMI","Smoking","AlcoholDrinking","Stroke","PhysicalHealth","MentalHealth","DiffWalking","Sex","AgeCategory","Race","Diabetic","PhysicalActivity","GenHealth","SleepTime","Asthma","KidneyDisease","SkinCancer"]]
y = df[["HeartDisease"]]


# In[12]:


x


# In[13]:


y


# In[14]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# In[15]:


scaler = StandardScaler()


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[16]:


param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2', 'elasticnet', 'none']
}

# Lojistik regresyon modeli
model = LogisticRegression()

# Grid Search oluştur
grid_search = GridSearchCV(model, param_grid, cv=5) 


# In[17]:


grid_search.fit(x_train_scaled, y_train)


# In[18]:


print("En iyi hiperparametreler:", grid_search.best_params_)
print("En iyi çapraz doğrulama skoru:", grid_search.best_score_)


# In[19]:


best_model = grid_search.best_estimator_


# In[20]:


best_model.fit(x_train_scaled, y_train)


# In[21]:


y_head = best_model.predict(x_test_scaled)


# In[22]:


print("Eğitim Başarısı : %",accuracy_score(y_test,y_head))


# In[2]:


import pickle


# In[ ]:


pickle.dump(best_model, open("logistic_Regression_model.pickle",'wb'))


# ### tekrar açıldığında importları ve bu satırları çalıştırmak yeterli. Sonrasında değer verilerek tahmin yapabilir

# In[3]:


best_model = pickle.load(open("logistic_Regression_model.pickle","rb"))


# In[ ]:




