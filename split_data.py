import csv
from sklearn.model_selection import train_test_split

def split(input_file_path, train_file_path, test_file_path):
    data = []
    labels = []
    with open(input_file_path, newline='') as f:
        reader = csv.reader(f, delimiter = ',')
        for row in reader:
            if row[3].split('/')[1] == "World":
                continue
            data.append(row[2])
            labels.append(row[3])
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.1, random_state = 42)
    print("{0} train documents".format(len(y_train)))
    print("{0} test documents".format(len(y_test)))
    with open(train_file_path, 'w') as f:
        writer1 = csv.writer(f, delimiter = ',')
        for i in range(len(X_train)):
            writer1.writerow(['None', 'None', X_train[i], y_train[i]])
    with open(test_file_path, 'w') as f:
        writer2 = csv.writer(f, delimiter = ',')
        for i in range(len(X_test)):
            writer2.writerow(['None', 'None', X_test[i], y_test[i]])

# def count_label(input_file_path):
#     with open(input_file_path, newline='') as f:
#         reader = csv.reader(f, delimiter = ',')
#         for row in reader:

class Node:
    def __init__(self, key, num):
        self.key = key
        self.child = {}
        self.num = num

def test_node(node):
    queue = [node]
    while queue:
        node = queue.pop(0)
        print("Node: " + node.key)
        print([(n.key, n.num) for n in node.child.values()])
        queue.extend(node.child.values())


def filtered(input_file_path, output_file_path, child_limit, level):
    print("build the tree")
    root = Node("Top", 0)
    cur = root
    with open(input_file_path, newline='') as fr:
        reader = csv.reader(fr, delimiter = ',')
        for row in reader:
            label = row[3]
            ls = label.split('/')[1:]
            if ls[0] not in {'Arts', 'Computers', 'Sports', 'Shopping'}:
                continue
            for i in range(min(level,len(ls))):
                if not cur.child or ls[i] not in cur.child.keys():
                    new_node = Node(ls[i],1)
                    cur.child[ls[i]] = new_node
                    cur = cur.child[ls[i]]

                else:
                    cur.child[ls[i]].num += 1
                    cur = cur.child[ls[i]]
                # print("Count: {0}".format(count))
                # print(min(4,len(ls)))
            cur = root

    print("refine the tree")
    queue = [root]
    while queue:
        node = queue.pop(0)
        child_values = list(node.child.values())
        child_values.sort(key= lambda x:x.num, reverse= True)
        node.child = {n.key: n for n in child_values[:child_limit]}
        queue.extend(node.child.values())

    test_node(root)
    print("output refined csv")

    with open(input_file_path, newline='') as fr, open(output_file_path, 'w') as fw:
        reader = csv.reader(fr, delimiter=',')
        writer = csv.writer(fw, delimiter=',')
        cur = root
        for row in reader:
            label = row[3]
            ls = label.split('/')[1:]
            flag = 0
            for i in range(min(level,len(ls))):
                if ls[i] in cur.child.keys():
                    cur = cur.child[ls[i]]
                else:
                    flag = 1
                    break
            if flag == 0:
                writer.writerow(row)
            cur = root

def get_FT_data(input_file_path, output_file_path):
    with open(input_file_path, 'r') as f1, open(output_file_path, 'w') as f2:
        reader = csv.reader(f1, delimiter=',')
        for row in reader:
            newline = '__label__' + '/'.join(row[3].split('/')[:5]) + " "
            newline += row[2]
            f2.write(newline + '\n')



if __name__ == '__main__':
    filtered('dmoz.csv', './level/filtered_8_level.csv', 6, 8)
    split('./level/filtered_8_level.csv','./level/train_8_level.csv', './level/test_8_level.csv')
    #get_FT_data('test.csv', 'FT_test.txt')