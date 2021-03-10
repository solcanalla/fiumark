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

	#Elimino volvería ya que es el training set
	y = df.volveria
	# Elimino categorías id_usuario, nombre por alta cardinalidad
	# Elimino fila ya que presenta un 20% de missings
	X = df.drop(columns=['volveria','nombre','fila','id_usuario'])
	return train_test_split(X, y, test_size=0.15, random_state=42)


def common_preprocessing(df):

	df_precio['categoria_precio'] = df_precio['precio_ticket'].apply(categoria)

	# Reemplazo normal por 2d
	df = df.replace({'tipo_de_sala':'normal'},'2d')

	# Reemplazo el nombre de la sede
	df = df.replace({'nombre_sede':'fiumark_palermo'},'Palermo')
	df = df.replace({'nombre_sede':'fiumark_chacarita'},'Chacarita')
	df = df.replace({'nombre_sede':'fiumark_quilmes'},'Quilmes')

	#Agrupo amigos y parientes en compañidea
	df["compañía"] = df["amigos"] + df["parientes"]

	#Encondeo sin orden
	columns_to_encode = ['genero','fila','nombre_sede','tipo_de_sala']
	df_to_encode = pd.DataFrame(df[columns_to_encode],columns=columns_to_encode)
	ohe = OneHotEncoder(drop='first').fit(df_to_encode.astype(str))
	column_name = ohe.get_feature_names(df_to_encode.columns)
	one_hot_encoded_frame =  pd.DataFrame(ohe.transform(df_to_encode.astype(str)).todense().astype(int), columns= column_name)

	del df['genero']
	del df['nombre_sede']
	del df['tipo_de_sala']
	del df['fila']
	del df['amigos']
	del df['parientes']
	
	df = pd.concat([df, one_hot_encoded_frame], axis=1)

	return df

def decisiontree_preprocessing(X):
	return common_preprocessing(X)

def knn_preprocessing(X):
	X = common_preprocessing(X) #Si normalizo tengo 0.7
	X = pd.DataFrame(MinMaxScaler().fit_transform(X), index=X.index, columns=X.columns)
	return X
