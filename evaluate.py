import csv
import numpy as np


def evaluate(pred_label, true_label, eval_type, level):
    pred_ls = pred_label.split('/')[1:]
    true_ls = true_label.split('/')[1:]
    res = []
    if eval_type == 'accu':
        for j in range(level):
            if j >= len(true_ls):
                res.append(1)
            elif j >= len(pred_ls):
                res.append(0)
            else:
                if '/'.join(pred_ls[:j+1]) == '/'.join(true_ls[:j+1]):
                    res.append(1)
                else:
                    res.append(0)
        assert len(res) == level
        return np.array(res)


def evaluate_accu(predict_label_path, true_label_path, level):
    predict_labels = []
    true_labels = []
    with open(true_label_path, 'r') as f1:
        reader = csv.reader(f1)
        for row in reader:
            true_labels.append(row[3])
    with open(predict_label_path, 'r') as f2:
        for row in f2:
            predict_labels.append(row[:-1])
    eval_all = np.array([0 for _ in range(level)])
    for i in range(len(predict_labels)):
        eval_all += evaluate(predict_labels[i], true_labels[i], 'accu', level)

    return eval_all/len(predict_labels)



#def evaluate_recall(predict_label_path, true_label_path):
if __name__ == '__main__':

    print('LR_deterministic       {0}'.format(evaluate_accu('test_LR_accu_deterministic.csv', 'test.csv')))
    print('FT_deterministic       {0}'.format(evaluate_accu('test_FT_accu_deterministic.csv', 'test.csv')))
    print('SVM_hierachical        {0}'.format(evaluate_accu('test_SVM_accu.csv', 'test.csv')))
    print('LR_probalistic         {0}'.format(evaluate_accu('test_LR_accu_prob.csv', 'test.csv')))
    print('FT_probalistic         {0}'.format(evaluate_accu('test_FT_accu_prob.csv', 'test.csv')))
    print('SVM_flat               {0}'.format(evaluate_accu('test_SVM_flat.txt', 'test.csv')))
    print('LR_flat                {0}'.format(evaluate_accu('test_LR_flat.txt', 'test.csv')))
    print('FT_hierachical_softmax {0}'.format(evaluate_accu('test_FT_hs.txt', 'test.csv')))


