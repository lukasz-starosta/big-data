from prepare.get_prepared_data import get_prepared_data
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from metrics.metrics import print_stats, get_roc_curve_plot, learning_curve_plot
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.ensemble import BalancedBaggingClassifier

data, labels = get_prepared_data()

X = data.iloc[:, :-1]
y = data['SUSP_SEX'].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=0, shuffle=True)

# X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)

models = [
    # GaussianNB(),
    # MultinomialNB(),
    # RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1),
    BalancedRandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=2),
    BalancedBaggingClassifier(
        base_estimator=HistGradientBoostingClassifier(random_state=42),
        n_estimators=100,
        random_state=42,
        n_jobs=2,
    )
]
model_labels = [model.__class__.__name__ for model in models]

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


# learning_curve_plot(models,
#                     model_labels, training_data=X_train,
#                     training_target=y_train, test_data=X_test, test_target=y_test)
