from sklearn.linear_model import LogisticRegression
from utils import *

def train_model(train_features, train_y):

    clf_rl = LogisticRegression(penalty="l1", solver='liblinear')
    clf_rl.fit(train_features,train_y)
    print("For the regularized logistic regression, the coefficents are as follows:\n")
    print(clf_rl.coef_)

    print("The training accuracy score is", clf_rl.score(train_features, train_y)*100, "%")
    return(clf_rl)


def main():
    train_features = load_features('/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/data/processed/train.npz')
    train_y = load_train_label('/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/data/processed/train_y.csv')

    class_names = train_y.label.unique()
    logistic_lasso_model = train_model(train_features, train_y)

    print('Evaluating logistic regression lasso regularized model...')
    evaluate_model(train_features, train_y, class_names, logistic_lasso_model )

    print("Saving model...")
    save_model(logistic_lasso_model , '/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/models/logistic_lasso.pkl')

if __name__ == '__main__':
    main()
