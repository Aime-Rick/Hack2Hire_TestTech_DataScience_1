import pickle
import numpy as np
import pandas as pd


with open("model.pkl", "rb") as file:
	classifier = pickle.load(file)

with open("processing.pkl", "rb") as file:
    preprocessing = pickle.load(file)

X=pd.DataFrame(np.array([25, 'male', '2', 'own', 'little', 'moderate', 6000, 40, 'car']).reshape(1,-1), columns=['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration', 'Purpose'])

X_processed = preprocessing.transform(X)
print(X_processed)
risk_prediction = classifier.predict(X_processed)[0]

if risk_prediction == 1:
    print("The person is a bad credit risk.")
else:
    print("The person is a good credit risk. ")

 


