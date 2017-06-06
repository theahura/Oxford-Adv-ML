"""
Amol Kapoor
Implementation of Q3 from ws2.
"""

import GPy
import matplotlib.pyplot as plt
import numpy as np
import pandas
from statsmodels.tsa.ar_model import AR

mgdata = pandas.read_csv('mgdata.dat', sep=' ', header=None, index_col=0)

data = np.array(mgdata[mgdata.columns[0]])
data = data.reshape((1201, 1))
time = np.array(range(len(data)))
time = time.reshape(1201, 1)

y_train = data[:1000]
y_test = data[1000:]
x_train = time[:1000]
x_test = time[1000:]

print y_test

# Autoregression
model = AR(y_train)
model_fit = model.fit(maxlag=100, ic='t-stat')
predictions = model_fit.predict(start=len(x_train),
                                end=len(x_train)+len(x_test)-1, dynamic=False)
print model_fit.k_ar
plt.plot(x_test, y_test, 'r-')
plt.plot(x_test, predictions, 'g-')
plt.title("Autoregressive Model")
plt.show()

# Poly Regression
poly_x_train = x_train.reshape(len(x_train))
poly_y_train = y_train.reshape(len(y_train))
z = np.poly1d(np.polyfit(poly_x_train, poly_y_train, 30))
pred_y_test = z(x_test)
prev_y_train = z(x_train)
prev_y_train = prev_y_train.reshape(len(x_train))
pred_y_test = pred_y_test.reshape(len(x_test))
plt.plot(time, data, 'r-')
plt.plot(x_train, prev_y_train,b'g-')
plt.plot(x_test, pred_y_test, 'b-')
plt.title("Polynomial Regression")
plt.show()

# GP
print x_train
k = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
m = GPy.models.GPRegression(x_train, y_train, k)
m.optimize_restarts(num_restarts=1)
fig = m.plot()
predictions = m.predict(x_test)[0]
predictions = predictions.reshape(len(x_test))

# Print out
plt.plot(x_test, y_test, 'r-')
plt.plot(x_test, predictions, 'b-')
plt.title("RBF Gaussian Process")
plt.show()
