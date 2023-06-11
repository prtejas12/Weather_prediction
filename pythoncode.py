import math
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
dataset=pd.read_csv("train_dataset.csv")
dataset.tail(15)
dataset.describe()
dataset.info()
dataset["Next_Tmax"]
total=6589-6567
total#total number of null values
dataset.dropna(subset=["Next_Tmax"],inplace=True)
x=dataset.drop("Next_Tmax",axis=1)
y=dataset["Next_Tmax"]
list=dataset.select_dtypes(include="float64").columns.tolist()
list.pop()
plt.figure(figsize=(12,9))
sns.heatmap(dataset.corr(),annot=True);

"""From the above plot we can observe that the Next_Tmax has high correlation with the below ploted features"""

sns.lineplot(data=dataset,x=x["Present_Tmax"],y=y)
plt.title("Present_Tmax vs Next_Tmax")

sns.lineplot(data=dataset,x="Present_Tmin",y=y)
plt.title("Present_Tmin vs Next_Tmax")

sns.boxplot(dataset["Next_Tmax"])

"""From the above box-plot we find that there are few outliers

<b>Fill all the null values with the respective mean of the respective columns</b>
"""

imputer=SimpleImputer(strategy="mean").fit(dataset[list])
x[list]=imputer.transform(x[list])
x.tail()
x.info()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
modellr=LinearRegression()
modellr.fit(x_train,y_train)
y_predlr=modellr.predict(x_test)
modelsvm=SVR()
modelsvm.fit(x_train,y_train)
y_pred=modelsvm.predict(x_test)
modeldt=DecisionTreeRegressor()
modeldt.fit(x_train,y_train)
y_predt=modeldt.predict(x_test)
print("SVR RMS=",math.sqrt(mean_squared_error(y_pred,y_test)))
print("LinearRegresssion RMS=",math.sqrt(mean_squared_error(y_predlr,y_test)))
print("DessionTree RMS=",math.sqrt(mean_squared_error(y_predt,y_test)))
