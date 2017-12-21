from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import csv
import numpy as np
import string


def vectorize(file_path):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(data_generater(file_path))
    return matrix, vectorizer


def data_generater(file_path):
    with open(file_path, newline='') as f:
        reader = csv.reader(f, delimiter = ',')
        for row in reader:
            yield row[2]


def load_text(file_path):
    with open(file_path, newline='') as f:
        reader = csv.reader(f, delimiter = ',')
        text = np.array([row[2] for row in reader])
        return text


def load_labels(file_path):
    with open(file_path, newline='') as f:
        reader = csv.reader(f, delimiter = ',')
        labels = np.array([row[3] for row in reader])
        return labels


def clean(text):
    table = str.maketrans('', '', string.punctuation)
    stripped = text.translate(table)
    return stripped.lower()


if __name__ == '__main__':
    print(clean("haham ,dfaj.dajfad"))


