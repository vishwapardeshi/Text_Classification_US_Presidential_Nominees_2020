import xgboost as xgb
from sklearn import model_selection, preprocessing
from utils import *

def train_model(train_features, train_y):
    m_xgb = xgb.XGBClassifier(objective='binary:logistic',booster = "gbtree", eval_metric='auc')
    clf_xgb = model_selection.GridSearchCV(m_xgb,{'max_depth': [4,6, 8],'n_estimators': [50,100,200]})

    clf_xgb.fit(train_features,train_y)
    print("The best score", clf_xgb.best_score_)

    print("The training accuracy score is", clf_xgb.score(train_features, train_y)*100, "%")
    return(clf_xgb)

def main():
    train_features = load_train_features('/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/data/processed/train.npz')
    train_y = load_train_label('/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/data/processed/train_y.csv')

    class_names = train_y.label.unique()
    xgboost_model = train_model(train_features, train_y)

    print('Evaluating logistic regression lasso regularized model...')
    evaluate_model(train_features, train_y, class_names, xgboost_model )

    print("Saving model...")
    save_model(xgboost_model , '/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/models/xgboost.pkl')

if __name__ == '__main__':
    main()
