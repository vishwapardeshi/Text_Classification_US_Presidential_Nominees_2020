import os
import pandas as pd

import sklearn
from utils import *

def main():
    dir_model = '/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/models/'
    clf_semi = load_model(os.path.join(dir_model, 'clf_semi.pkl'))

    test_df = load_data('/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/data/processed/test_df.csv')
    test_features = load_features('/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/data/processed/test.npz')

    predicted_labels = clf_semi.predict(test_features)

    file_label = test_df.file_name.tolist()

    #convert to a dataframe
    final_submission = pd.DataFrame({'FILE': file_label, 'MODEL1':part_2_labels, 'MODEL2': part_3_labels})
    final_submission.head()
    #sorting by file name
    submission_df = final_submission.sort_values('FILE')
    final_submission.to_csv('submission.txt', sep='\t', index=False)
