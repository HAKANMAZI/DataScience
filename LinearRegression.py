ice = pd.read_csv("IceCreamData.csv")
X = ice[["Revenue"]]
y = ice["Temperature"] 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression(fit_intercept=True)
regressor.fit(X_train, y_train)
print('Linear Model Coefficient (m): ', regressor.coef_)
print('Linear Model Coefficient (b): ', regressor.intercept_)
#y = b + mx
y_predict = regressor.predict(X_test)

plt.scatter(X_train, y_train, color = 'blue')
plt.plot(X_train, regressor.predict(X_train), color = 'red')
plt.ylabel('Revenue [dollars]')
plt.xlabel('Temperature [degC]')
plt.title('Revenue vs. Temperature @Ice Cream Temp-Revenue Project(Training dataset)')

y_predict = regressor.predict([[687]])
