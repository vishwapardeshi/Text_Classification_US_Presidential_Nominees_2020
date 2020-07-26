import pandas as pd
import re
import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import model_selection, preprocessing
from process_files import save_data

def load_data(filepath):
    df = pd.read_csv(filepath)
    return(df)

def label_encode(train_df):
    # label encode the target variable
    train_y = train_df.label
    encoder = preprocessing.LabelEncoder()
    train_y_vector = encoder.fit_transform(train_y)
    return(train_y)

def clean_data(quote):
    my_stopwords = nltk.corpus.stopwords.words('english')
    my_stopwords.extend(["america", "american", "united", "people"])
    my_punctuation = '!"$%&\'()*+,-.…/:;<=>?[\\]^_`{|}~•@’'
    #print(quote)
    quote = quote.lower() # lower case
    #print(quote)
    quote = quote.strip()#remove double spacing
    #quote_new = quote.translate(str.maketrans(dict.fromkeys(string.punctuation)))
    quote = "".join([char.lower() for char in quote if char not in my_punctuation])
    quote = re.sub('['+my_punctuation + ']+', ' ', quote) # strip punctuation
    quote = " ".join([word for word in quote.split(' ') if word not in my_stopwords])
    return quote.strip()


def lemmatize_data(quote):
    lemmatizer = WordNetLemmatizer()
    tk = WhitespaceTokenizer()
    quote = " ".join([lemmatizer.lemmatize(word) for word in tk.tokenize(quote)])
    return(quote)

def main():
    train_df = load_data('/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/data/interim/train.csv')
    test_df = load_data('/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/data/interim/test.csv')

    train_df['clean_quote'] = train_df.Quotes.apply(clean_data)
    train_df["final_clean"] = train_df.clean_quote.apply(lambda x: lemmatize_data(x))

    train_y = label_encode(train_df)
    print("Save the data to data/interim")
    save_data(train_df, '/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/data/interim/clean_train.csv')
    save_data(test_df, '/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/data/interim/clean_test.csv')
    save_data(train_y, '/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/data/processed/train_y.csv')

if __name__ == '__main__':
    main()
