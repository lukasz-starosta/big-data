from prepare.get_prepared_data import get_prepared_data
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data, labels = get_prepared_data()

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.7, random_state=0)
classifier = GaussianNB()
y_pred = classifier.fit(X_train, y_train).predict(X_test)

print(f'Accuracy: {accuracy_score(y_pred=y_pred, y_true=y_test)}')
