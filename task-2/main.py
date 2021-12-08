from prepare.get_prepared_data import get_prepared_data
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from metrics.metrics import print_stats, get_roc_curve_plot, learning_curve_plot
import matplotlib.pyplot as plt

data, labels = get_prepared_data()

X = data.iloc[:, :-1]
y = data['SUSP_SEX'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=0, shuffle=True)

decision_tree = DecisionTreeClassifier()
gaussian_nb = GaussianNB()
bernoulli_nb = BernoulliNB()
kneighbors = KNeighborsClassifier()
forest = RandomForestClassifier()

models = [decision_tree, gaussian_nb, bernoulli_nb, kneighbors, forest]
model_labels = ['Decision Tree', 'Gaussian NB',
                'Bernoulli NB', 'KNeighbors', 'Forest']

for i in range(len(models)):
    print(model_labels[i])

    y_pred = models[i].fit(X_train, y_train).predict(X_test)
    print_stats(target_values=y_test, predicted_values=y_pred,
                labels=labels, model_label=model_labels[i])

    fig, ax = plt.subplots()
    x, y, _ = get_roc_curve_plot(models[i], X_test, y_test)
    ax.plot(x, y, label=f'ROC {model_labels[i]}')
    ax.legend()
    fig.savefig(f'fig {model_labels[i]}')


learning_curve_plot(models,
                    model_labels, training_data=X_train,
                    training_target=y_train, test_data=X_test, test_target=y_test)
