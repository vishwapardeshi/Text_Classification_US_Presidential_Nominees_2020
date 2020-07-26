from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import string
from scipy.sparse import hstack
from scipy import sparse

def load_data(filepath):
    df = pd.read_csv(filepath)
    return(df)

def vectorize_data(train_df, test_df):
    vectorizer = CountVectorizer(max_df=0.95, min_df=13, ngram_range=(1, 1))
    # apply transformation
    train_x = vectorizer.fit_transform(train_df['final_clean'])
    tf_feature_names = vectorizer.get_feature_names()

    test_x = vectorizer.transform(test_df.Quotes)
    return(train_x, test_x)

def feature_engineering(train_df, test_df):

    train_df['char_count'] = train_df['Quotes'].apply(len)
    train_df['word_count'] = train_df['Quotes'].apply(lambda x: len(x.split()))
    train_df['word_density'] = train_df['char_count']/(train_df['word_count'] + 1)
    train_df['punctuation_count'] = train_df['Quotes'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation)))
    train_df['title_word_count'] = train_df['Quotes'].apply(lambda x: len([word for word in x.split() if word.istitle()]))
    train_df['upper_case_word_count'] = train_df['Quotes'].apply(lambda x: len([word for word in x.split() if word.isupper()]))

    test_df['char_count'] = test_df['Quotes'].apply(len)
    test_df['word_count'] = test_df['Quotes'].apply(lambda x: len(x.split()))
    test_df['word_density'] = test_df['char_count']/(test_df['word_count']+1)
    test_df['punctuation_count'] = test_df['Quotes'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation)))
    test_df['title_word_count'] = test_df['Quotes'].apply(lambda x: len([word for word in x.split() if word.istitle()]))
    test_df['upper_case_word_count'] = test_df['Quotes'].apply(lambda x: len([word for word in x.split() if word.isupper()]))

    num_features = [f_ for f_ in train_df.columns\
                    if f_ in ["char_count", "word_count", "word_density", 'punctuation_count','title_word_count', 'upper_case_word_count']]

    for f in num_features:
        all_cut = pd.cut(pd.concat([train_df[f], test_df[f]], axis=0), bins=20, labels=False, retbins=False)
        train_df[f] = all_cut.values[:train_df.shape[0]]
        test_df[f] = all_cut.values[train_df.shape[0]:]

    train_num_features = train_df[num_features].values
    test_num_features = test_df[num_features].values

    return(train_num_features, test_num_features, train_df, test_df)

def save_data(matrix,filepath):
    """Save dataframe to the filepath as csv"""
    sparse.save_npz(filepath, matrix)

def save_dataframe(df,filepath):
    """Save dataframe to the filepath as csv"""
    df.to_csv(filepath, index = False)

def main():
    train_df = load_data('/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/data/interim/clean_train.csv')
    test_df = load_data('/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/data/interim/clean_test.csv')

    train_x, test_x, = vectorize_data(train_df, test_df)

    train_num_features, test_num_features, train_df, test_df = feature_engineering(train_df, test_df)

    train_features = hstack([train_x, train_num_features])
    test_features = hstack([test_x, test_num_features])
    print(type(train_features))

    print("Saving features of train & test data")
    save_data(train_features, '/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/data/processed/train.npz')
    save_data(test_features, '/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/data/processed/test.npz')

    save_dataframe(train_df, '/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/data/processed/train_df.csv')
    save_dataframe(test_df, '/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/data/processed/test_df.csv')


if __name__ == '__main__':
    main()
