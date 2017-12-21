from preprocessing import vectorize, load_labels, load_text, clean
from fastText import train_supervised
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import numpy as np
import time
from evaluate import evaluate_accu
import csv


class Node:
    def __init__(self, key, inp, l):
        """
        The key is the label at each level. Ex: Top/Sports, top and sports are keys at first/second level respectively.
        The classifier is the corresponding classifier of this node.
        The input is index list of the training examples of this particular node.
        The child is the list of the children nodes of this node.
        key: String
        classifier: Classifier()
        input: list
        child: dictionary with Node as the value
        """
        self.key = key
        self.classifier = None
        self.input = inp
        self.child = {}
        self.pre_node = None
        self.score = 0
        self.level = l


class Tree:
    def __init__(self, classifier_type, input_file_path, lev):
        """
        Build the tree
        :param classifier_type: String ("LR", "SVM", "FT")
        :param input_file_path: The path for the data file
        """
        print("Building the tree")
        self.root = Node("Top", [], 0)
        self.root.score = 1
        self.tree_type = classifier_type
        self.input_file_path = input_file_path
        print("Loading the labels")
        self.labels = load_labels(input_file_path)
        if classifier_type == "FT":
            print("Start loading data")
            self.text = load_text(input_file_path)
        else:
            print("Start vectorizing")
            self.matrix, self.vectorizer = vectorize(input_file_path)
            # joblib.dump(vectorizer, "./load_model/" + 'vectorizerc_leadp.pkl')
            # scipy.sparse.save_npz("./load_model/" + 'lp_matrix.npz', matrix)
            # with open("./load_model/" + 'vectorizerc_leadp.pkl', 'rb') as f:
            #     self.vectorizer = pickle.load(f)
            # self.matrix = scipy.sparse.load_npz("./load_model/" + 'lp_matrix.npz')
        cur = self.root
        for i, label in enumerate(self.labels):
            if not label:
                continue
            levels = label.split("/")[1:]
            if len(levels) > 0 and levels[0] == 'World':
                continue
            for j, key in enumerate(levels):
                # ignore the first "Top" taxonomy
                if j >= lev:
                    continue
                child_keys = [c for c in cur.child.keys()]
                if not child_keys:
                    other_node = Node("*", [], cur.level + 1)
                    cur.child["*"] = other_node
                    other_node.pre_node = cur
                if key not in child_keys:
                    new_node = Node(key, [i], cur.level + 1)
                    cur.child[key] = new_node
                    new_node.pre_node = cur
                    cur.input.append(i)
                    cur = cur.child[key]
                else:
                    cur.input.append(i)
                    cur = cur.child[key]
            if "*" in cur.child.keys():
                cur.child["*"].input.append(i)
                # if the label doesn't stop at the leaves of the tree. Add it to the OTHER node of the curNode's child.
            cur = self.root
            if i % 200000 == 0:
                print("{0} documents are processed".format(i))

    def traverse(self):
        """

        :return: List. The level order traversal of the nodes in the tree.
        """
        queue = [self.root]
        traverse_list = []
        while queue:
            node = queue.pop(0)
            # dequeue
            traverse_list.append(node)
            queue.extend(node.child.values())
        return traverse_list

    def train(self):
        """
        the train prorosess is:
        1. traverse of the tree
        2. train the model in each Node, where the train data is the child nodes' input attribute(list of input index)
           and the label is the child nodes' keys.
        :return: None
        """
        print("start training!!")
        traverse_list = self.traverse()
        self.train_model(traverse_list)

    def train_model(self, traverse_list):
        """
        This is used for LR and FT model as each step of the model will output a probability.
        :param traverse_list:
        :return: None
        """
        for i, node in enumerate(traverse_list):
            print("Node key: " + node.key)
            print("Child keys:")
            print(node.child.keys())
            print("Trained {0} models".format(i))
            if len(node.child.keys()) <= 2:
                continue
            if node.key == "*":
                continue
            if not node.child:
                continue
            train_data = self.get_train_data(node)
            if self.tree_type == "FT":
                node.classifier = train_supervised(input=train_data)
            elif self.tree_type == "LR":
                node.classifier = LogisticRegression(multi_class='multinomial', solver='newton-cg')
                node.classifier.fit(train_data[0], train_data[1])
            elif self.tree_type == "SVM":
                node.classifier = LinearSVC()
                node.classifier.fit(train_data[0], train_data[1])

    def get_train_data(self, node):
        """
        get the data for this node from the text list or the vectorizer.
        :param node:
        :return:
        """
        if self.tree_type == "FT":
            node_txt = self.text[node.input]
            node_labels = self.labels[node.input]
            with open("./temp.txt", "w") as f:
                for i, sentence in enumerate(node_txt):
                    new_sentence = sentence
                    ls = node_labels[i].split('/')
                    if node.level < len(ls) -1:
                        f.write("__label__" + node_labels[i].split("/")[node.level+1] + " " + clean(sentence) + "\n")
                    else:
                        f.write("__label__" + "* " + sentence + "\n")
            return "./temp.txt"
        else:
            node_labels = self.labels[node.input]
            next_level_labels = []
            for label in node_labels:
                ls = label.split('/')
                if node.level < len(ls) - 1:
                    next_level_labels.append(ls[node.level + 1])
                else:
                    next_level_labels.append('*')
            return self.matrix[node.input], next_level_labels

    def get_test_data(self, input_path):
        test_text = load_text(input_path)
        test_label = load_labels(input_path)
        if self.tree_type == "FT":
            test_text2 = np.array([clean(t) for t in test_text])
            return test_text2, test_label

        else:
            test_text = self.vectorizer.transform(test_text)
            return test_text, test_label

    def predict(self, test_input_path, k, predict_type):
        """

        :param test_input_path:
        :param k:
        :param predict_type:
        :return:
        """
        test_data, test_label = self.get_test_data(test_input_path)
        if predict_type == 'deterministic':
            final_labels = self.predict_deterministic(test_data)
        else:
            final_labels = []
            for i, line in enumerate(test_data):
                k_nodes = self.predict_prob(line, k)
                k_nodes.sort(key=lambda x: x.score, reverse=True)
                max_node = k_nodes[0]
                full_path = ""
                assert isinstance(max_node, Node)
                cur = max_node
                while cur.pre_node:
                    full_path = "/" + cur.key + full_path
                    cur = cur.pre_node
                full_path = "Top" + full_path
                final_label = full_path
                final_labels.append(final_label)
                if i % 1000 == 0:
                    print("predict {0} test documents".format(i))

        return final_labels

    def predict_prob(self, data, _k=5):
        queue = [self.root]
        final_nodes = []
        while queue:
            node = queue.pop(0)
            if len(node.child.keys()) == 2:
                for k in node.child.keys():
                    if k != "*":
                        node.child[k].score = node.score
                queue.extend(node.child.values())
                continue
            if not node.child:
                final_nodes.append(node)
                queue.extend(node.child.values())
                continue
            if self.tree_type == "FT":
                node_labels, node_probs = node.classifier.predict(data, k=_k)
                node_labels = [n[9:] for n in node_labels]
            else:
                predicted_probs = node.classifier.predict_proba(data)
                index = np.argsort(predicted_probs.reshape(-1))[::-1][:_k]
                node_labels = node.classifier.classes_[index]
                node_probs = predicted_probs.reshape(-1)[index]
            # assign new node score to next level nodes
            for i in range(len(node_labels)):
                node.child[node_labels[i]].score = node_probs[i] * node.score
            queue.extend(node.child.values())
        return final_nodes

    def predict_deterministic(self, data):
        cur = self.root
        final_labels = []
        for i, line in enumerate(data):
            final_label = ""
            while cur.child:
                if len(cur.child.keys()) == 2:
                    for k in cur.child.keys():
                        if k != "*":
                            final_label += cur.key + "/"
                            cur = cur.child[k]
                    continue
                final_label += cur.key + "/"
                label = cur.classifier.predict(line)
                if self.tree_type == "FT":
                    f_label = label[0][0][9:]
                    cur = cur.child[f_label]
                else:
                    cur = cur.child[label[0]]
            final_label += cur.key
            final_labels.append(final_label)
            cur = self.root
            if i % 1000 == 0:
                print("predict {0} test documents".format(i))
        return final_labels


if __name__ == '__main__':
    t = Tree("LR", "./level/train_8_level.csv", 3) # !!!
    # print([c.key for c in t.traverse()])

    a = time.time()
    t.train()
    # res = t.root.classifier.predict_proba(t.vectorizer.transform(["Includes a filmography, biography and links"]))
    # print(res)
    r = t.predict('./level/test_8_level.csv', 5, 'deterministic')
    # res = t.predict("../test.csv", 3, 'deterministic')
    with open('./temp.csv', 'w') as f:
        for l in r:
            f.write(l)
            f.write("\n")
    b = time.time()
    print('3 levels') # !!!
    print("Hierarchical LR")
    print('Time used: {0}'.format(b-a))
    print("Accuracy: {0}".format(evaluate_accu('temp.csv', './level/test_8_level.csv', 3))) # !!!
