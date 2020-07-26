from scipy import sparse
import pandas as pd
import joblib

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def load_features(filepath):
    matrix = sparse.load_npz(filepath)
    return matrix

def load_train_label(filepath):
    label = pd.read_csv(filepath)
    return label

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def save_model(model, model_filepath):
    joblib.dump(model, open(model_filepath, 'wb'))

def load_model(model_filepath):
    model = joblib.load(model_filepath)
    return model

def evaluate_model(train_features, train_y, class_names, model):
    model_report = classification_report(train_y, model.predict(train_features))

    matrix = confusion_matrix(train_y, model.predict(train_features), labels = class_names)
    model_acc = matrix.diagonal()/matrix.sum(axis=1 )
    model_df = pd.DataFrame({'accuracy': model_acc}, index=class_names)
    confusion_matrix_df = pd.DataFrame(matrix, index = class_names, columns = class_names)

    print(model_report)
    print("=========================================================\n\n\nThe accuracy for each class is\n\n\n", model_df)
    print("=========================================================\n\n\nConfusion Matrix\n\n")
    confusion_matrix_df.head(12)
