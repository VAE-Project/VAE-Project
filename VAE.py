import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.layers import Input,Dense,Lambda
from tensorflow.keras.models import Model
from tensorflow import keras
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler

df = pd.read_csv('../Data/vehicle.csv')
df=df[df['VEHYEAR']>=2017]
df
