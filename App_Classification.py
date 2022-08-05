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
print(model.predict([290.4551412,0.144164196,0.053058009,74.63506664,1189.545213,15.93815436,35.70333099,65.54132446,330.2672929]))




