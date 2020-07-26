import os
import pandas as pd

import sklearn
from sklearn import model_selection, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from utils import *

#get top instances
def getTopInstances(clf_semi, test, testFeature, total):
    prob_predicted = {}
    k = 0
    for i, row in test.iterrows():
        prob_dict = dict(zip(clf_semi.classes_, clf_semi.predict_proba(testFeature)[k]))
        results = list(map(lambda x: (x[0], x[1]), sorted(zip(clf_semi.classes_, clf_semi.predict_proba(testFeature)[k]), key=lambda x: x[1], reverse=True)))
        prob_predicted[i] = results[0]
        k += 1
    print("The class and their associated probability", prob_predicted)
    sort_prob = sorted(prob_predicted.keys(), key=lambda x: prob_predicted[x][1], reverse=True)[:total]
    return(sort_prob, prob_predicted)

#get top 10 index and classes
def addTopTest(test, train, prob_class, prob_pred):
  #find label class and index to add
    k = 1
    l = 11
    for i in prob_class:
        curr_row = []
        #remove from the test features
        curr_s= test.loc[[i]]
        #drop these rows from the data frame
        test = test.drop([i])
        clean = cleanQuotes(curr_s.Quotes.item())
        final_clean = lemmatizeQuotes(clean)
        curr_row.append([prob_pred[i][0], curr_s.Quotes.item(), clean, final_clean, curr_s.char_count.item(), curr_s.word_count.item(), curr_s.word_density.item(),\
                         curr_s.punctuation_count.item(), curr_s.title_word_count.item(), curr_s.upper_case_word_count.item()])
        #add curr_row to training df
        train = train.append(pd.DataFrame(curr_row, columns = train.columns))
        print("Adding row ",i, " ----- ", k, "/", l)
        k += 1
    #print("Adding row\n",curr_row)
    #print("\nShape after adding row", train.shape)
    return(train, test)

def generateFeatures(train, test):
    semi_train_x = vectorizer.transform(train['final_clean'])
    semi_test_x = vectorizer.transform(test.Quotes)
    semi_features = [f_ for f_ in train.columns\
                   if f_ in ["char_count", "word_count", "word_density", 'punctuation_count','title_word_count', 'upper_case_word_count']]
    print("The shape of train:", semi_train_x.shape, "\ntest:", semi_train_x .shape)


    for f in semi_features:
        all_cut = pd.cut(pd.concat([train[f], test[f]], axis=0), bins=20, labels=False, retbins=False)
        train[f] = all_cut.values[:train.shape[0]]
        test[f] = all_cut.values[train.shape[0]:]

    train_dup_features = train[semi_features].values
    test_dup_features = test[semi_features].values
    dup_train_features = hstack([semi_train_x, train_dup_features])
    dup_test_features = hstack([semi_test_x, test_dup_features])

    return(dup_train_features, dup_test_features)

def main():
    dir_model = '/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/models/'
    clf_logistic_lasso = load_model(os.path.join(dir_model, 'logistic_lasso.pkl'))
    clf_semi = sklearn.base.clone(clf_logistic_lasso, safe=True)

    train_df = load_data('/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/data/processed/train_df.csv')
    test_df = load_data('/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/data/processed/test_df.csv')

    test_features = load_features('/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/data/processed/test.npz')

    dup_train = train_df
    dup_test = test_df


    pc, pp = getTopInstances(clf_semi, dup_test, test_features, 11)

    #generate new train and test
    dup_train, dup_test = addTopTest(dup_test, dup_train, pc, pp)

    #generate feature for new test train
    curr_train_feature, curr_test_feature = generateFeatures(dup_train, dup_test)

    clf_semi.fit(curr_train_feature,dup_train.label)

    accuracy = []
    ac = clf_semi.score(train_features, train_df.label)
    accuracy.append(ac)
    print("The accuracy of the model after 1st iteration on the updated train data is" , clf_semi.score(curr_train_feature, dup_train.label) * 100, "%" )
    print("The accuracy of the model after 1st iteration on the original train data is" ,ac * 100, "%" )


    for ix in range(2, 11):
        print("===============================" , ix, "===================================")
        if ix == 10:
            pc, pp = getTopInstances(clf_semi, dup_test, curr_test_feature, 12)
        else:
            pc, pp = getTopInstances(clf_semi, dup_test, curr_test_feature, 11)
        #generate new train and test
        dup_train, dup_test = addTopTest(dup_test, dup_train, pc, pp)
        print("The shape of train:", dup_train.shape, "\ntest:", dup_test.shape)
        #generate feature for new test train
        curr_train_feature, curr_test_feature = generateFeatures(dup_train, dup_test)
        clf_semi.fit(curr_train_feature,dup_train.label)
        ac = clf_semi.score(train_features, train_df.label)
        accuracy.append(ac)
        print("The accuracy of the model after", ix," iteration on the update train data is" , clf_semi.score(curr_train_feature, dup_train.label) * 100, "%" )
        print("The accuracy of the model after", ix,"iteration on the original train data is" ,ac * 100, "%" )

    print("Saving model...")
    save_model(clf_semi, '/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/models/clf_semi.pkl')


if __name__ == '__main__':
        main()
