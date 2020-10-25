# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # FiuMark - Análisis de datos
#
# *Encuestas realizadas durante 2 meses sobre los clientes de FiuMark que fueron a ver Frozen 3.* 
#

import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport


# +
# Traemos los datos de los 2 archivos y los agrupamos en un solo dataframe.

df1 = pd.read_csv('https://drive.google.com/uc?export=download&id=1i-KJ2lSvM7OQH0Yd59bX01VoZcq8Sglq')
df2 = pd.read_csv('https://drive.google.com/uc?export=download&id=1km-AEIMnWVGqMtK-W28n59hqS5Kufhd0')

df = df1.merge(df2,on='id_usuario', how='left')

# -

# Usando pandas profiling
report = ProfileReport(df, title='Fiumark', minimal=True)
report

# Como primer análiss sobre el profiling se puede ver que no hay usuarios que hayan completados 2 veces la encuesta, esto nos habla de que no es necesario eliminar duplicados.
#
# Por otro lado identificamos 2 columnas que vamos a eliminar para seguir con el análisis ya que no aportan información relveante para este análisis: nombre y id_ticket.

df.sample(50)


