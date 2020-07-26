# Which US Democratic Presidential Nominee said this? Warren? Biden? Sanders?
### Text Classification of quotes from candidates vying to be the Democratic presidential nominee for the 2020 US presidential election.
Here, all data has been extracted from debates between candidates. I have built a NLP classification model to identify who said what for a subset of unlabeled data.

## Methodology
The quotes are subjected to basic text-preprocessing steps such as
1. Stopword removal

2. Punctuation removal

3. Lemmatization

4. Tokenization using unigram

To prepare data for modeling, I performed feature engineering. Here, I engineered features which utilize count of various components of the text such as character, word, punctuation etc.

**The text classification is done using Supervised & Semi-Supervised techniques.**
The following models were explored:
1. Regularized Logistic Regression

2. Random Forest

3. XGBoost

## Tools & Technology
```
1. NLP: nltk, TfidfVectorizer, CountVectorizer
2. ML: sklearn, xgboost, scipy
3. Visualization: Seaborn, Matplotlib
4. Exploration: Jupyter Notebooks
```

