#File for preprocessing data
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
	OneHotEncoder,
	MinMaxScaler
)


def get_dataset():
	df1 = pd.read_csv('https://drive.google.com/uc?export=download&id=1i-KJ2lSvM7OQH0Yd59bX01VoZcq8Sglq')
	df2 = pd.read_csv('https://drive.google.com/uc?export=download&id=1km-AEIMnWVGqMtK-W28n59hqS5Kufhd0')
	df = df1.merge(df2,on='id_usuario', how='left')
	return df

def get_train_test_data(df):
	y = df.volveria
	X = df.drop(columns='volveria')
	return train_test_split(X, y, test_size=0.20, random_state=42)

def common_preprocessing(df):
	df.reset_index(inplace=True,drop=True)
	
	#Clean
	del df['nombre']

	#Missing values
	#fila 80%
	#nombre_sede 0.2%

	#Missing values
	#edad 20%
	edad_filled = SimpleImputer(strategy='median').fit_transform(df[['edad']])
	filled = pd.DataFrame(edad_filled).add_prefix('edad_')
	df = pd.concat([df, filled], axis=1)
	del df['edad']

	#Categoricas con baja cardinalidad
	#genero, nombre de sede, tipo de sala y fila.

	#Encondeo sin orden
	ohe = OneHotEncoder(drop='first').fit(df[['genero','fila','nombre_sede','tipo_de_sala']].astype(str))
	matrix_result = ohe.transform(df[['genero','fila','nombre_sede','tipo_de_sala']].astype(str)).todense().astype(int)
	df = pd.concat([df, pd.DataFrame(matrix_result)], axis=1)

	del df['genero']
	del df['nombre_sede']
	del df['tipo_de_sala']
	del df['fila']

	#Categóricas de alta cardinalidad
	del df['id_ticket']

	return df

def decisiontree_preprocessing(X):
	return common_preprocessing(X)

def knn_preprocessing(X):
	X = common_preprocessing(X) #Si normalizo tengo 0.7
	X = pd.DataFrame(MinMaxScaler().fit_transform(X), index=X.index, columns=X.columns)
	return X




