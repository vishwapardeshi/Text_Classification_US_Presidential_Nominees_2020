import pandas as pd
import numpy as np
import re
import string

from os import listdir
from os.path import isfile, join



def read_files(path):
    """ Read files from folder"""
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    name_list, file_list = [], []
    print("Reading in ", len(onlyfiles), " files.")
    for f in onlyfiles:
        file = open(path + "/" + f, 'r')
        lines  = file.readlines()
        file_list.append(lines)
        name_list.append(f.split("_")[:2])
    print("Read ", len(name_list), " files.")
    return(name_list,file_list)


def mismatch_training(names, train_df):
    """Check if file name is different than the target label"""
    count = 0
    candidate_name = train_df.label
    for i in range(len(names)):
      c = candidate_name[i].lower().strip()
      #print(names[i], c)
      if(names[i][0] == 'O'):
          names[i][0] = names[i][0] + names[i][1]
      if (names[i][0].lower().replace("'", "") != c):
        print(names[i][0], c)
        count += 1
    print("There are", count, "mismatch in training files")
    return(count)

def load_train(base_path):
    """ Load training data from files into a DataFrame"""
    #reading train data
    train_path = base_path + "train"
    names, file_content = read_files(train_path)

    #convert to dataframe
    #convert to data frame
    train_all = pd.DataFrame(file_content, columns = ['c'])
    split_data = train_all["c"].str.split(":")
    #extract all names
    c_name = []
    c_quote = []
    for s in split_data:
      c_name.append(s[0].split()[1].upper().replace("'", ""))
      c_quote.append(str(s[1:]))

    train_df = pd.DataFrame({"label" : c_name, "Quotes": c_quote})

    #perform sanity check
    count = mismatch_training(names, train_df)
    if count != 0:
        raise ValueError("Incorrect File Names!")
        return

    return(train_df)

def load_test(base_path):
    """ Load testing data from files into a DataFrame"""
    #reading test data
    test_path = base_path + "test"
    test_file, test_list = read_files(test_path)

    new_file = []
    for f_ in test_file:
      new_file.append("_".join(f_))
    #test_file

    test_df = pd.DataFrame({"file_name" : new_file,"Quotes": test_list})
    test_df.head()

    test_df['Quotes'] = test_df['Quotes'].apply(lambda x: ','.join(map(str, x)))
    return(test_df)

def save_data(df,filepath):
    """Save dataframe to the filepath as csv"""
    df.to_csv(filepath, index = False)

def main():
    base_path = '/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/data/raw/'
    train_df = load_train(base_path)
    #There are 528 training observations and 111 test observations
    test_df = load_test(base_path)

    print("Save the data to data/interim/")
    save_data(train_df, '/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/data/interim/train.csv')
    save_data(test_df, '/Users/vishwapardeshi/Documents/GitHub/Text_Classification_US_Presidential_Nominees_2020/data/interim/test.csv')

if __name__ == '__main__':
    main()
