from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report 
from utils import *

def train_model(train_features, train_y):
    clf_rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=100, max_leaf_nodes=None, min_impurity_split=None,
                min_samples_leaf=3, min_samples_split=10, n_estimators=10, random_state=None)

    clf_rf.fit(train_features,train_y)

    print("The training accuracy score is", clf_rf.score(train_features, train_y)*100, "%")
    return(clf_rf)

def main():
    train_features = load_train_features('/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/data/processed/train.npz')
    train_y = load_train_label('/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/data/processed/train_y.csv')

    class_names = train_y.label.unique()
    random_forest_model = train_model(train_features, train_y)

    print('Evaluating random forest model...')
    evaluate_model(train_features, train_y, class_names, random_forest_model)

    print("Saving model...")
    save_model(random_forest_model, '/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/models/random_forest.pkl')

if __name__ == '__main__':
    main()
