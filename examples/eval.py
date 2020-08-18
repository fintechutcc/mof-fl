from numpy import load
from keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report

# laod data
data = load('examples/data/mnist/random/data_party0.npz')

lst = data.files 

X_train, y_train, X_test, y_test = data[lst[0]], data[lst[1]], data[lst[2]], data[lst[3]]

# load model
model = load_model('model_1597593575.0589483.h5') #Name of your model should have different number from here

# summarize model
print(model.summary())

# Add one more dimension to X_test to match the input data
X_test = np.expand_dims(X_test, axis=-1)

# Prediction
pred = model.predict_classes(X_test)

# Evaluate the performance
print(classification_report(y_test, pred))