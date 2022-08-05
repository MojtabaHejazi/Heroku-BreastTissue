import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import pickle

dataset = pd.read_csv('BreastTissue.csv')

x= dataset.iloc[:, :9]

y= dataset.iloc[:, -1]

KNC=KNeighborsClassifier()

KNC.fit(x,y)

pickle.dump(KNC, open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))




