# -*- coding: utf-8 -*-

import pandas as pd
from google.colab import files

# Upload the CSV file from your local machine
uploaded = files.upload()

# Assuming you upload one file, get its name
for fn in uploaded.keys():
  print(f'User uploaded file "{fn}"')
  file_name = fn

df = pd.read_csv(file_name)
df = df.drop(columns=["PassengerId"])

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix


X=df.drop(columns=["Survived"])
Y=df["Survived"]

x_train , x_test , y_train , y_test = train_test_split(
    X,Y,test_size=0.2,random_state=42
)

Scaler = StandardScaler()
x_train = Scaler.fit_transform(x_train)
x_test = Scaler.transform(x_test)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

y=y_pred = model.predict(x_test)


accuracy = accuracy_score(y_test , y_pred)
conf_metrix = confusion_matrix(y_test , y_pred)


print("accuracy:", accuracy)
print("confusion_metrix\n",conf_metrix)
