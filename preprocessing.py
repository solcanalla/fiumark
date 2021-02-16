#File for preprocessing data
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
	LabelEncoder,
	OneHotEncoder,
	Normalizer,
	MinMaxScaler
)


def get_dataset():
	df1 = pd.read_csv('https://drive.google.com/uc?export=download&id=1i-KJ2lSvM7OQH0Yd59bX01VoZcq8Sglq')
	df2 = pd.read_csv('https://drive.google.com/uc?export=download&id=1km-AEIMnWVGqMtK-W28n59hqS5Kufhd0')
	df = df1.merge(df2,on='id_usuario', how='left')
	return df

def get_train(df):
	y = df.volveria
	X = df.drop(columns='volveria')
	return X,y

def get_train_test_data(df):
	y = df.volveria
	X = df.drop(columns='volveria')
	return train_test_split(X, y, test_size=0.15, random_state=42)

def common_preprocessing(df):
	#Reseteo el indice por si entro con el train 
	df.reset_index(inplace=True,drop=True)
	
	#Clean
	del df['nombre']

	#Missing values
	#edad 20%
	#fila 80%
	#nombre_sede 0.2%

	median_imputer = SimpleImputer(strategy='median')
	edad_filled = median_imputer.fit_transform(df[['edad']])
	filled = pd.DataFrame(edad_filled).add_prefix('edad_')
	df = pd.concat([df, filled], axis=1)
	del df['edad']

	#Categoricas con baja cardinalidad
	#genero, nombre de sede, tipo de sala y fila.

	#Encondeo sin orden
	ohe = OneHotEncoder(drop='first')
	
	genero_encoded = (
	    ohe.fit_transform(df[['genero']].astype(str)).todense().astype(int)
	)
	genero_encoded = pd.DataFrame(genero_encoded).add_prefix('male_')
	df = pd.concat([df, genero_encoded], axis=1)

	sede_encoded = (
	    ohe.fit_transform(df[['nombre_sede']].astype(str)).todense().astype(int)
	)
	sede_encoded = pd.DataFrame(sede_encoded).add_prefix('sede_')
	df = pd.concat([df, sede_encoded], axis=1)

	sala_encoded = (
	    ohe.fit_transform(df[['tipo_de_sala']].astype(str)).todense().astype(int)
	)
	sala_encoded = pd.DataFrame(sala_encoded).add_prefix('sala_')
	df = pd.concat([df, sala_encoded], axis=1)

	fila_encoded = (
	    ohe.fit_transform(df[['fila']].astype(str)).todense().astype(int)
	)
	fila_encoded = pd.DataFrame(fila_encoded).add_prefix('fila_')
	df = pd.concat([df, fila_encoded], axis=1)

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
	X = common_preprocessing(X)
	scaler = MinMaxScaler() #Si normalizo tengo un 0.7
	X = scaler.fit_transform(X)
	return X
