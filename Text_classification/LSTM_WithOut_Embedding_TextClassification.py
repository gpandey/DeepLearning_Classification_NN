 #import sys
# sys.path.append('C:\\Users\\gitaa\\Desktop\\gita\\')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#ignore warning
import warnings
warnings.filterwarnings('ignore')

from sklearn.utils import shuffle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Embedding,Flatten,LSTM,SpatialDropout1D,Conv1D,MaxPooling1D,Dropout
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from  keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import re

#Scikit-learn packages
from sklearn.manifold import TSNE

#Reading the file
data_path = r"C:\Users\gitaa\Desktop\DataScienceProject\NLP_project\Multi_class_classification\train\train.csv"
df = pd.read_csv(data_path)
df_te = pd.read_csv( r"C:\Users\gitaa\Desktop\DataScienceProject\NLP_project\Multi_class_classification\test\test.csv")

#understanding the data
print(df.head())
print("\n shape of train: \n",df.shape)
print(df.info())
print('\nPrinting the value: \n')
print(df.author.value_counts(dropna=False))

data = df.drop(['author'], axis = 1)

data = data['text'].values

MAX_NB_WORDS = 10000
tokenizer = Tokenizer(nb_words= MAX_NB_WORDS)
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)
v2 = len(tokenizer.word_index)+1

#printing the sequences
print(len(sequences))

word_index = tokenizer.word_index
print('Found %s unique tokens.'% len(word_index))
v2 = len(tokenizer.word_index)+1

MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 100

data1 = pad_sequences(sequences,maxlen= MAX_SEQUENCE_LENGTH)
print(data1.shape)
labels = pd.get_dummies(df['author']).values

print('Shape of data tensor:', data1.shape)
print('Shape of label tensor:', labels.shape)

#Train test data
x_train, x_test,y_train, y_test = train_test_split(data1,labels,test_size=0.22,random_state=50)
print(x_train.shape,y_train.shape)
print(x_test.shape, y_test.shape)

#Modeling LSTM
model = Sequential()
model.add(Embedding(MAX_NB_WORDS,EMBEDDING_DIM,input_length =data1.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100,dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss ='categorical_crossentropy',optimizer='rmsprop',metrics =['accuracy'])
print(model.summary())

epochs = 100
batch_size = 16

history = model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_split=0.2)
accr = model.evaluate(x_test,y_test)
print('Test set \n Loss: {:0.3f}\n Accuracy: {:0.3f}'.format(accr[0],accr[1]))
