import pandas as pd
import numpy as np
import tensorflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras import backend as K
from tensorflow.keras.layers import Dropout
df = pd.read_csv('output.csv')
df = df.dropna()
brand_counts = df['Brand'].value_counts()

brands_to_keep = brand_counts[brand_counts >= 6].index.tolist()

df = df[df['Brand'].isin(brands_to_keep)]
categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
df_dummy = pd.get_dummies(df, columns = categorical_columns)
df = df.merge(df_dummy, left_index = False, right_index = False)

df = df.drop(columns = categorical_columns)
req = df["Price"]
ar = list(req)
for i in range(len(ar)):
    ar[i] = float(ar[i])
del df["Price"]
df["Price"] = pd.DataFrame(ar)
X = df.drop(columns = ['Price'])
Y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25)
np.random.seed(123)

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

def create_model(learn_rate = 0.001, activation = 'relu', neurons = 128):
    model = Sequential()
    model.add(Dense(neurons, input_dim = len(X_train.columns), kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(1))
    opt = keras.optimizers.Adam(learning_rate = learn_rate)
    model.compile(loss = root_mean_squared_error, optimizer = 'adam')
    return model

model = KerasRegressor(build_fn = create_model, verbose = 0)
callbacks = [keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 15, verbose = 1)]

history = model.fit(X_train, y_train, epochs = 500, batch_size = 1, validation_split = 0.25, verbose = 0, callbacks = callbacks)
X = X_test.iloc[1].values
X = X.reshape(1, -1)

predicted_price = model.predict(X)
print(predicted_price)