# coding=utf-8
from nltk.classify import maxent
from get_train_test_data import split_dataset
from feature_extractor import FEATURE_NAMES
from common import *


def train_and_test(all_dataset, test_prop, foldnum, predict_result_file_path, test_record_file_path):
    print_function_info("train_and_test")

    predict_result_file = open(predict_result_file_path, "w")
    test_record_file = open(test_record_file_path, "w")  # 每次训练预测的结果(for multiple fold train/test)

    accumulated_accuraty = 0.0

    for i in range(foldnum):
        print "foldnum:"+str(i)
        train_set, test_set = split_dataset(all_dataset, test_prop, foldnum, i)

        train_data = [single_data.data for single_data in train_set]
        test_data = [single_data.data for single_data in test_set]
        test_text = [single_data.text for single_data in test_set]

        encoding = maxent.TypedMaxentFeatureEncoding.train(train_data, count_cutoff=3, alwayson_features=True)
        classifier = maxent.MaxentClassifier.train(train_data, bernoulli=False, encoding=encoding, trace=0)
        classifier.show_most_informative_features(10)
        predict_results = classifier.classify_many([feature_vec for feature_vec, label in test_data])

        save_predict_detail(predict_result_file, test_text, test_data, predict_results, i)
        y_accuracy, n_accuracy, y_recall, n_recall, total_accuracy = calculate_performance(test_data, predict_results)

        test_record_file.write("Yaccuracy:" + str(y_accuracy) + "\n")
        test_record_file.write("Naccuracy:" + str(n_accuracy) + "\n")
        test_record_file.write("Yrecall:" + str(y_recall) + "\n")
        test_record_file.write("Nrecall:" + str(n_recall) + "\n")
        test_record_file.write("totalAccuracy:" + str(total_accuracy) + "\n")
        accumulated_accuraty += total_accuracy
        test_record_file.write("#"*20 + "\n")

        print "#"*30

    predict_result_file.close()
    test_record_file.close()
    print "mean totalAccuracy:", accumulated_accuraty / foldnum


def save_predict_detail(predict_result_file, test_text, test_data, predict_results, fold_index):
    # write title
    predict_result_file.write("foldIndex,oritext,real_label,predict_label,isRight")
    for feature_name in FEATURE_NAMES:
        predict_result_file.write(out_file_splitter + feature_name)
    predict_result_file.write("\n")

    # write data and predict result
    for text, (feature_vec, real_label), result in zip(test_text, test_data, predict_results):
        predict_result_file.write(str(fold_index) + out_file_splitter)
        predict_result_file.write(text.encode('utf-8') + out_file_splitter)
        predict_result_file.write(real_label.encode('utf-8') + out_file_splitter)
        predict_result_file.write(result.encode('utf-8') + out_file_splitter)
        predict_result_file.write(str(real_label == result))
        for feature_name in FEATURE_NAMES:
            try:
                predict_result_file.write(out_file_splitter + str(feature_vec[feature_name]).encode("utf-8"))
            except:
                predict_result_file.write(out_file_splitter + str(feature_vec[feature_name].encode("utf-8")))
                predict_result_file.write("\n")


def calculate_performance(test_data, predict_results):
    right_y = 0
    wrong_y = 0
    right_n = 0
    wrong_n = 0

    for t, r in zip(test_data, predict_results):
        if t[1] == "y":
            if r == "y":
                right_y += 1
            else:
                wrong_n += 1
        else:
            if r == "y":
                wrong_y += 1
            else:
                right_n += 1

    try:
        y_accuracy = right_y / float(right_y + wrong_y)
    except:
        y_accuracy = -0.1
    try:
        n_accuracy = right_n / float(right_n + wrong_n)
    except:
        n_accuracy = -0.1
    try:
        y_recall = right_y / float(right_y + wrong_n)
    except:
        y_recall = -0.1
    try:
        n_recall = right_n / float(right_n + wrong_y)
    except:
        n_recall = -0.1
    try:
        total_accuracy = (right_y + right_n) / float(right_y + right_n + wrong_y + wrong_n)
    except:
        total_accuracy = -0.1

    print "totalAccuracy:", total_accuracy

    return y_accuracy, n_accuracy, y_recall, n_recall, total_accuracy
