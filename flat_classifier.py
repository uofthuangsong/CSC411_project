from preprocessing import load_text, load_labels, clean, vectorize
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from sklearn import preprocessing
from fastText import train_supervised
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import sys
sys.path.append('/malex2/NYT_annotated_corpus/xu/Song411')
import time
from evaluate import evaluate_accu


def classify(level):
    a = time.time()
    train_labels = load_labels('./level/train_8_level.csv') # !!!

    test_text = load_text('./level/test_8_level.csv') # !!!
    test_labels = load_labels('./level/test_8_level.csv') # !!!

    train_labels = ['/'.join(label.split('/')[:level+1]) for label in train_labels]
    test_labels = ['/'.join(label.split('/')[:level+1]) for label in test_labels]
    print(train_labels[0])
    assert len(train_labels[0].split('/')) == level + 1
    le = preprocessing.LabelEncoder()
    le.fit(np.concatenate((train_labels, test_labels)))
    encoded_train_labels = le.transform(train_labels)
    encoded_test_labels = le.transform(test_labels)
    print('number of labels: ')
    print(len(le.classes_))
    print(encoded_train_labels.shape)
    print(encoded_test_labels.shape)
    print('load complete')


    # svm = OneVsRestClassifier(LinearSVC())
    lg = OneVsRestClassifier(LogisticRegression())

    m, vectorizer = vectorize('./level/train_8_level.csv') # !!!
    print('vectorize complete')
    # sup = train_supervised(input='FT_train.txt')

    # svm.fit(m, encoded_train_labels)
    lg.fit(m, encoded_train_labels)
    print("fit complete")
    # print(svm.predict(vectorizer.transform(test_text[0])))
    # print(lg.predict(vectorizer.transform(test_text[0])))
    label_predict = lg.predict(vectorizer.transform(test_text))
    label_predict = le.inverse_transform(label_predict)
    # label_predict = []
    # for text in test_text:
    #     p, prob = sup.predict(clean(text))
    #     label_predict.append(p[0][9:])
    #     print(p)
    with open('temp.txt', 'w') as f:
        for pred in label_predict:
            f.write(pred + '\n')

    b = time.time()
    print('{0} levels:'.format(level))
    print('number of labels {0}'.format(len(le.classes_)))
    print('flat LR')
    print('Time used: {0}s'.format(b-a))
    print('Accuracy: ')
    print(evaluate_accu('temp.csv', './level/test_8_level.csv', level)) # !!!

if __name__ == '__main__':
    classify(3)