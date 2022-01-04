from sklearn import metrics
import numpy
from sklearn.model_selection import learning_curve
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as pyplot
import warnings


def get_accuracy(target_values, predicted_values, digits):
    return round(metrics.accuracy_score(
        target_values,
        predicted_values), digits)


def get_confusion_matrix(target_values, predicted_values):
    return metrics.confusion_matrix(
        target_values,
        predicted_values)


def get_metrics(target_values, predicted_values, labels, digits):
    return metrics.classification_report(
        target_values,
        predicted_values,
        target_names=list(map(str, labels)),
        digits=digits
    )


def get_precision(target_values, predicted_values, digits):
    return round(metrics.precision_score(
        target_values,
        predicted_values), digits)


def get_recall(target_values, predicted_values, digits):
    return round(metrics.recall_score(
        target_values,
        predicted_values), digits)


def get_roc_curve_plot(model, test_data, target_values):
    warnings.filterwarnings('ignore')
    roc_curve_plot = metrics.plot_roc_curve(
        model,
        test_data,
        target_values)

    fpr = roc_curve_plot.fpr
    tpr = roc_curve_plot.tpr
    roc_auc = roc_curve_plot.roc_auc

    return fpr, tpr, roc_auc


def print_confusion_matrix(target_values, predicted_values, model_label):
    matrix = metrics.confusion_matrix(
        target_values,
        predicted_values)

    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[
                                  'Female', 'Male'])
    disp.plot()
    pyplot.savefig(f'conf {model_label}')


def print_metrics(target_values, predicted_values, digits):
    tn, fp, fn, tp = get_confusion_matrix(
        target_values, predicted_values).ravel()

    metrics_names = ['Precision', 'Accuracy', 'Recall', 'Specificity']
    metrics_scores = [get_precision(target_values, predicted_values, digits),
                      get_accuracy(target_values, predicted_values, digits),
                      get_recall(target_values, predicted_values, digits),
                      round((tn/(tn+fp)), digits)]

    metrics_first_line = '  '
    metrics_second_line = '  '

    for name, score in zip(metrics_names, metrics_scores):
        length = max(len(name), len(str(score))) + 3
        metrics_first_line += name.ljust(length)
        metrics_second_line += str(score).ljust(length)

    print('\nMetrics:')
    print(metrics_first_line)
    print(metrics_second_line)


def print_stats(target_values, predicted_values, labels, model_label, digits=3):

    print('\nLabels: ' + str(labels))

    print_confusion_matrix(target_values, predicted_values, model_label)
    print_metrics(target_values, predicted_values, digits)


def learning_curve_add_subplot(model, label, training_data, test_data, training_target, test_target, ax):
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        numpy.concatenate((training_data, test_data)),
        numpy.concatenate((training_target, test_target)),
        train_sizes=[0.6, 0.7, 0.8, 0.9]
    )
    ax[0].plot(
        train_sizes,
        train_scores.mean(1),
        label=label
    )
    ax[1].plot(
        train_sizes,
        test_scores.mean(1),
        label=label
    )


def learning_curve_plot(models, labels, training_data, test_data, training_target, test_target):
    fig, ax = pyplot.subplots(1, 2, sharey='all', figsize=(9, 6))

    for i in range(len(models)):
        learning_curve_add_subplot(
            models[i], labels[i], training_data, test_data, training_target, test_target, ax)

    ax[0].set_title('Training', fontsize='medium')
    ax[1].set_title('Validation', fontsize='medium')
    ax[0].set_xlabel('Train size')
    ax[1].set_xlabel('Train size')
    ax[0].set_ylabel('Score')
    ax[1].tick_params(length=0)
    # ax[0].legend(prop={'size': 7})
    ax[1].legend()
    ax[0].grid(alpha=0.2)
    ax[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig('LC', bbox_inches='tight', dpi=300)
