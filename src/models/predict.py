
#### 16. Generate labels for semi-supervised

---
"""

part_3_labels = clf_semi.predict(test_features)
print(part_3_labels)

#sanity check
print("The length of the predicted values ", len(part_3_labels))
#extract file name from test_df
file_label = test_df.file_name.tolist()
print(file_label)

#convert to a dataframe
final_submission = pd.DataFrame({'FILE': file_label, 'MODEL1':part_2_labels, 'MODEL2': part_3_labels})
final_submission.head()

#sorting by file name
submission_df = final_submission.sort_values('FILE')

submission_df.head()

#CONVERT TO TXT FILE
final_submission.to_csv('submission.txt', sep='\t', index=False)
