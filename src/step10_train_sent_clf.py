import pickle
import os
import random
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from step9_get_sent_dataset import load_dataset

epoch = 3

# divide train/validation/test sets
# file_num = len(os.listdir(r"../ivan_data/merged"))
file_num = 75
train_ids = list(range(0, 100)) + list(range(140, 140))
valid_ids = list(range(100, 140))
# train_ids = list(range(file_num))
# vaild_and_test_ids = random.sample(train_ids, int(0.06 * file_num))
# train_ids = [i for i in train_ids if i not in vaild_and_test_ids]
# valid_ids = random.sample(vaild_and_test_ids, int(0.03 * file_num))
# test_ids = [i for i in vaild_and_test_ids if i not in valid_ids]


def main(train=True):

    if train:
        _, train_data, train_labels = zip(*load_dataset(train_ids))
        _, valid_data, valid_labels = zip(*load_dataset(valid_ids))
        _, test_data, test_labels = zip(*load_dataset(valid_ids))
        # _, test_data, test_labels = zip(*load_dataset(test_ids))

        vectorizer = DictVectorizer()
        vectorizer.fit(itertools.chain(train_data, valid_data, test_data))
        X, y = vectorizer.transform(train_data), train_labels
        X_valid, y_valid = vectorizer.transform(valid_data), valid_labels

        # clf = LogisticRegression(
        #     random_state=0, solver='lbfgs', multi_class='multinomial', class_weight="balanced", verbose=1, max_iter=100)

        clf = LogisticRegression(
            penalty='l2', dual=False, tol=0.0001, C=0.7, fit_intercept=True, intercept_scaling=1, class_weight="balanced", 
            random_state=0, solver='lbfgs', max_iter=125, multi_class='multinomial', verbose=1, warm_start=False, n_jobs=None)

        # X = X.toarray()
        # X_valid = X_valid.toarray()
        # clf = GaussianNB(priors=None, var_smoothing=1e-09)

        # clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,0
        #     decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
        #     max_iter=-1, probability=True, random_state=None, shrinking=True,
        #     tol=0.001, verbose=False)
            
        # print("==================== Epoch {} ==================".format(i))
        clf.fit(X, y)
        print("training set accuracy: %.3f%%" % (100. * clf.score(X, y)))
        print("validation set accuracy: %.3f%%" % (100. * clf.score(X_valid, y_valid)))
        print(classification_report(y, clf.predict(X)))
        print(classification_report(y_valid, clf.predict(X_valid)))

        pipeline = Pipeline([
                ("dict_vec", vectorizer), 
                ("clf", clf)])

        with open("clf_sent.pk", "wb") as f:
            pickle.dump(pipeline, f)
    
    # else:
    #     _, test_data, test_labels = load_dataset(test_ids)
        
    #     y_ = pipeline.predict(train_data)
    #     y_prob_ = pipeline.predict_proba(train_data) 
        

if __name__ == "__main__":
    main(train=True)