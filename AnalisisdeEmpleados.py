import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df=pd.read_csv('empleados.csv',delimiter=";")



# Imputar los valores nulos en la columna 'Salario' con la media de esa columna
df['Salario'] = df['Salario'].fillna(df['Salario'].mean())
# Imputar los valores nulos en la columna 'Salario' con la media de esa columna
df['Experiencia'] = df['Experiencia'].fillna(df['Experiencia'].mean())

x=df['Salario'].values.reshape(-1,1)
y=df['Experiencia'].values.reshape(-1,1)


regresion_lineal= LinearRegression()
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.20)
regresion_lineal.fit(x_train,y_train)

print('coeficiente de regresion: ', regresion_lineal.coef_)
print('Puntaje de regresion: ', regresion_lineal.score(x_test,y_test))
# data = pd.DataFrame({'k1': ['one', 'two'] * 3 +['two'],'k2': [1, 1, 2, 3, 3, 4, 4]})
# print(data)