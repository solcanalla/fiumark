#File for preprocessing data
from sklearn.model_selection import train_test_split

def get_train_test_data():
	df1 = pd.read_csv('https://drive.google.com/uc?export=download&id=1i-KJ2lSvM7OQH0Yd59bX01VoZcq8Sglq')
	df2 = pd.read_csv('https://drive.google.com/uc?export=download&id=1km-AEIMnWVGqMtK-W28n59hqS5Kufhd0')
	df = df1.merge(df2,on='id_usuario', how='left')
	
	y = df.volveria
	X = df.drop(columns='volveria')
	return train_test_split(X, y, test_size=0.15, random_state=42)

