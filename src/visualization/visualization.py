import seaborn as sns
import matplotlib.pyplot as plt

def compare_accuracy(model1, model2, model3, class_names):
    """
    create graph comparing accuracy and f1 score accross different class.
    """
    x = class_names
    plt.plot(model3_df.index, model3_df.accuracy)
    plt.plot(model1_df.index, model1_df.accuracy)
    plt.plot(model2_df.index, model2_df.accuracy)

    plt.title("Class-wise Accuracy")
    plt.legend(['XGBoost', 'Logistic', 'Random Forest'], loc='lower left')
    plt.xticks(rotation=90)
    plt.show()

def classification_accuracy(accuracy):
    """
    Classification Accuracy vs % test data
    """
    plt.axes()
    plt.plot(np.arange(10, 110, 10), accuracy)
    plt.xlim([0, 120])
    plt.ylim([0.95, 1])
    plt.yticks(np.arange(0.95, 1, 0.01))
    plt.xticks(np.arange(10, 110, 10))

    plt.title("Classification Accuracy vs % test data")

    plt.xticks(rotation=90)
    plt.xlabel("Percentage of test data added to the train set")
    plt.ylabel("Classification accuracy of original train set")
    plt.show()
    plt.show()
