==============================================================================
Final project for INFO 256, Applied Natural Language Processing, spring 2019.

Extract Information from Legal Texts (Court Decisions).

Ivan Liao, Kaifei Peng

==============================================================================
Instruction for use:

  All texts have been prepossessed and all intermediate models/supporting data have been set up.

  To extract metadata (i.e. Court names, plaintiffs and defendants, etc.) with regular expressions, run "step0".

  To check tfidf method for extracting patent terms and key sentences, run "step2".

  To check logistic regression for extracting patent terms, run "step4" & "step5". If you wanna change the training set and validation set, do following:
  1) open "step4.py", then modify "train_ids" and "valid_ids"; 
  2) open "step5.py", then modify the range of for statement beneath the line 'if __name__ == "__main__:"'.

  To check logistic regression for extracting key sentences, run "step10" & "step11". If you wanna change the training set and validation set, do following: 
  1) open "step10.py", then modify "train_ids" and "valid_ids"; 
  2) open "step11.py", then modify the range of for statement beneath the line 'if __name__ == "__main__:"'.

To check CNN for extracting key sentences, run "step7" & "step8". If you wanna change the training set and validation set, do following: 
  1) open "step7.py", then modify "train_ids" and "valid_ids"; 
  2) open "step8.py", then modify the range of for statement beneath the line 'if __name__ == "__main__:"'.

  If you want to start from the very beginning (i.e. start from preprocessing the texts), run steps [0, 1, 2, 4, 5, 7, 8, 10, 11]. 
  *WARNING: THIS WILL CLEAR OUT ALL MANUALLY ANNOTATED KEY SENTENCES LABELS IN "preprocessed/sents/*.txt", YOU MAY HAVE TO ENTER THIS DIRECTORY AND ANNOTATE THE SENTENCES AGAIN!
  
  If you are informed that glove word embedding file is missing, you can download "glove.42B.300d.50K.w2v" from the glove's website and save it as "glove.42B.300d.50K.w2v.txt" under the directory "ivan_data"
  
==============================================================================
ACKNOWLEDGEMENTS:
  Most of our code for CNN and calculating tfidf are downloaded from the github repo of anlp19: https://github.com/dbamman/anlp19 and modified to fit our work. This saves us plenty of time.
