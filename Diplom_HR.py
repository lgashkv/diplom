#!/usr/bin/env python
# coding: utf-8

# # Дипломная работа по курсу "Python для анализа данных"

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


df = pd.read_csv('C:\\Users\\Ольга\\Downloads\\HR.csv')
df


# In[3]:


df.info()


# ###### Рассчитайте основные статистики для переменных (среднее,медиана,мода,мин/макс,сред.отклонение).

# In[4]:


df.describe()


# In[5]:


df.mode()


# ###### Рассчитайте и визуализировать корреляционную матрицу для количественных переменных. Определите две самые скоррелированные и две наименее скоррелированные переменные. 

# In[6]:


df[['number_project', 'average_montly_hours']].corr(method='spearman')


# In[7]:


df[['last_evaluation', 'number_project']].corr(method='spearman')


# In[8]:


df[['last_evaluation', 'left']].corr(method='spearman')


# In[12]:


df[['number_project', 'Work_accident']].corr(method='spearman')


# In[14]:


from pylab import rcParams
rcParams['figure.figsize'] = 14,5
df.plot(kind='scatter', 
        x='number_project', 
        y='average_montly_hours', 
        title='Корреляция между количеством проэктов и средним количеством часов на работе в месяц')


# In[16]:


df.plot(kind='scatter', 
        x='number_project', 
        y='last_evaluation', 
        title='Корреляция между количеством проэктов и last_evaluation')


# In[17]:


sns.pairplot(df)


# ###### Рассчитайте сколько сотрудников работает в каждом департаменте. 

# In[18]:


df['department'].value_counts()


# ###### Показать распределение сотрудников по зарплатам. 

# In[19]:


salary  = df.groupby('salary')['salary'].count()
salary


# In[20]:


import matplotlib.pyplot as plt
ax = salary.plot(kind='barh', stacked=True),plt.title('Распределение сотрудников по зарплатам')
plt.xticks(rotation=60, horizontalalignment='right',fontsize=12)
ax


# ###### Показать распределение сотрудников по зарплатам в каждом департаменте по отдельности 

# In[21]:


salary_department  = df.groupby(['department','salary'])['salary'].count()
salary_department


# In[22]:


s_d = df[['department','salary']].groupby(by='department').count().sort_values(by='salary', ascending=False).head(10)
ax = salary_department.plot(kind='barh'),plt.xlabel('Salary'),plt.title('Распределение сотрудников по зарплатам в каждом департаменте')
ax


# ###### Проверить гипотезу, что сотрудники с высоким окладом проводят на работе больше времени,чем сотрудники с низким окладом 

# In[23]:


df_gipoteza = df[['average_montly_hours', 'salary']]
dfg  = df_gipoteza.groupby('salary')['average_montly_hours'].mean()
dfg


# In[24]:


import seaborn as sns  
ax = sns.boxplot(x='salary',y='average_montly_hours',data=df_gipoteza, color='#99c2a2')
plt.show()


# In[25]:


high = df_gipoteza.loc[df_gipoteza['salary'] =='high']
low = df_gipoteza.loc[df_gipoteza['salary'] =='low']
medium  = df_gipoteza.loc[df_gipoteza['salary'] =='medium']
from scipy.stats import f_oneway
F, p = f_oneway(high['average_montly_hours'], medium['average_montly_hours'],low['average_montly_hours'])  

alpha = 0.01  
print(F, p)   
if p > alpha:
    print('Одинаковое распределение (не отвергаем H0)')
else:
    print('Разное распределение (отклоняем H0)')


# ###### Гипотеза о том, что более высокооплачиваемые работники больше работают, неверна

# ###### Рассчитать следующие показатели среди уволившихся и не уволившихся сотрудников (по отдельности):                                                                                                                                                                              Доля сотрудников с повышением за последние 5 лет,                                                                                                    Средняя степень удовлетворенности                                                                                                                                    Среднее количество проектов 

# In[45]:


left = df.loc[df['left'] == 0]
noleft = df.loc[df['left'] == 1]


# In[46]:


left50 = left.loc[(left['time_spend_company'] < 6) & (left['promotion_last_5years'] == 0)]
noleft50 = noleft.loc[(noleft['time_spend_company'] < 6) & (noleft['promotion_last_5years'] == 0)]
left51 = left.loc[(left['time_spend_company'] < 6) & (left['promotion_last_5years'] == 1)]
noleft51 = noleft.loc[(noleft['time_spend_company'] < 6) & (noleft['promotion_last_5years'] == 1)]


# In[47]:


print('Доля сотрудников с повышением за последние 5 лет среди уволенных')
print(left50['promotion_last_5years'].count()/ df['promotion_last_5years'].count() * 100, "%")
print('Доля сотрудников с повышением за последние 5 лет среди работающих')
print(noleft50['promotion_last_5years'].count() / df['promotion_last_5years'].count() * 100, "%")


# In[48]:


print('Средняя степень удовлетворенности уволенных')
print(left['satisfaction_level'].mean())
print('Средняя степень удовлетворенности работающих')
print(noleft['satisfaction_level'].mean())


# In[49]:


print('Среднее количество проектов уволенных')
print(left['number_project'].mean())
print('Среднее количество проектов работающих')
print(noleft['number_project'].mean())


# ###### Разделить данные на тестовую и обучающую выборки                                                                                Построить модель LDA, предсказывающую уволился ли сотрудник на основе имеющихся факторов (кроме department и salary)                                                                                                                                                                   Оценить качество модели на тестовой выборки

# In[50]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[51]:


X = df[['left']]
y = df[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years']]


# In[52]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.06, random_state=42)


# In[53]:


X_train.shape


# In[54]:


y_train.shape


# In[55]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[56]:


model.coef_


# In[57]:


model.intercept_


# In[58]:


model.score(X_test, y_test)


# In[44]:


y_pred = model.predict(X_test)
y_pred


# In[ ]:





# In[ ]:




