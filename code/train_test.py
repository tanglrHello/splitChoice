# coding=utf-8
import time
import os
from nltk.classify import maxent
from get_train_test_data import split_dataset
from feature_extractor import FEATURE_NAMES
from common import *
from post_process import *


class TrainAndTest:
    def __init__(self, all_dataset, test_prop, foldnum,
                 predict_files_dir_path, record_file_path,
                 y_prop_in_trainset=False):
        self.all_dataset = all_dataset
        self.test_prop = test_prop
        self.foldnum = foldnum
        self.y_prop_in_trainset = y_prop_in_trainset

        self.predict_files_dir_path = predict_files_dir_path
        self.predict_res_file = None
        self.record_file_path = record_file_path

        self.y_precisions = []
        self.n_precisions = []
        self.y_recalls = []
        self.n_recalls = []
        self.total_precisions = []

        self.train_test_id = None
        self.post_processor = PostProcessor()

        self.reset_result_records()

    def reset_result_records(self):
        self.y_precisions = []
        self.n_precisions = []
        self.y_recalls = []
        self.n_recalls = []
        self.total_precisions = []

    def train_and_test(self):
        print_function_info("train_and_test")
        self.train_test_id = time.strftime('%Y-%m-%d-%H-%I-%M-%S', time.localtime(time.time()))

        self.open_predict_res_file()
        for i in range(self.foldnum):
            print "foldnum:"+str(i)
            train_set, test_set = split_dataset(self.all_dataset, self.test_prop,
                                                self.foldnum, i, self.y_prop_in_trainset)

            train_data = [single_data.data_for_train_test for single_data in train_set]
            test_data = [single_data.data_for_train_test for single_data in test_set]
            test_text = [single_data.ori_text for single_data in test_set]

            encoding = maxent.TypedMaxentFeatureEncoding.train(train_data, count_cutoff=3, alwayson_features=True)
            classifier = maxent.MaxentClassifier.train(train_data, bernoulli=False, encoding=encoding, trace=0)
            classifier.show_most_informative_features(10)
            predict_results = classifier.classify_many([feature_vec for feature_vec, label in test_data])

            post_predict_results = self.post_processor.post_process(test_set, test_text, predict_results)

            self.save_predict_detail(test_text, test_data, predict_results, post_predict_results, i)
            self.record_and_show_performance(test_data, post_predict_results)

        self.predict_res_file.close()

        self.post_processor.show_post_process_effect()
        self.show_performance()
        self.save_train_test_settings_and_performance()

    def open_predict_res_file(self):
        if not os.path.exists(self.predict_files_dir_path):
            os.mkdir(self.predict_files_dir_path)

        self.predict_res_file = open(self.predict_files_dir_path + self.train_test_id + ".csv", "w")

    def save_predict_detail(self, test_text, test_data, predict_results, post_predict_results, fold_index):
        # write title
        self.predict_res_file.write("foldIndex,oritext,real_label,predict_label,post_predict_results,isRight")
        for feature_name in FEATURE_NAMES:
            self.predict_res_file.write(out_file_splitter + feature_name)
        self.predict_res_file.write("\n")

        # write data and predict result
        for text, (feature_vec, real_label), result, post_result \
                in zip(test_text, test_data, predict_results, post_predict_results):
            self.predict_res_file.write(str(fold_index) + out_file_splitter)
            self.predict_res_file.write(text.encode('gbk') + out_file_splitter)
            self.predict_res_file.write(real_label.encode('gbk') + out_file_splitter)
            self.predict_res_file.write(result.encode('gbk') + out_file_splitter)
            self.predict_res_file.write(post_result.encode('gbk') + out_file_splitter)
            self.predict_res_file.write(str(real_label == result))
            for feature_name in FEATURE_NAMES:
                try:
                    self.predict_res_file.write(out_file_splitter + str(feature_vec[feature_name]).encode("gbk"))
                except:
                    self.predict_res_file.write(out_file_splitter + str(feature_vec[feature_name].encode("gbk")))
            self.predict_res_file.write("\n")

    def record_and_show_performance(self, test_data, predict_results):
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

        self.y_precisions.append(y_accuracy)
        self.n_precisions.append(n_accuracy)
        self.y_recalls.append(y_recall)
        self.n_recalls.append(n_recall)
        self.total_precisions.append(total_accuracy)

        print "totalAccuracy:", total_accuracy
        print "n_recall:", n_recall
        print "y_recall:", y_recall

    def save_train_test_settings_and_performance(self):
        record_file = open(self.record_file_path, "a")
        record_file.write("train_test id:" + self.train_test_id + "\n")
        record_file.write(self.get_settings_info())
        record_file.write(self.get_performance_info())
        record_file.write(self.post_processor.get_post_process_info())
        record_file.write("#"*60 + "\n")
        record_file.close()

    def get_settings_info(self):
        settings_info = ""
        settings_info += "test prop:" + str(self.test_prop) + "\n"
        settings_info += "y_prop_in_trainset:" + str(self.y_prop_in_trainset) + "\n"

        settings_info += "post_to_n_funcs1:"
        for func in self.post_processor.post_to_n_funcs1:
            settings_info += str(func)
        settings_info += "\npost_to_n_funcs2:"
        for func in self.post_processor.post_to_n_funcs2:
            settings_info += str(func)
        settings_info += "\n"
        settings_info += "features:  " + FEATURE_NAMES + "\n"

        return settings_info

    def get_performance_info(self):
        performance_info = ""
        performance_info += "y_precisions:" + self.get_mean_and_all_performance_string(self.y_precisions) + "\n"
        performance_info += "n_precisions:" + self.get_mean_and_all_performance_string(self.n_precisions) + "\n"
        performance_info += "y_recalls:" + self.get_mean_and_all_performance_string(self.y_recalls) + "\n"
        performance_info += "n_recalls:" + self.get_mean_and_all_performance_string(self.n_recalls) + "\n"
        performance_info += "total_precisions:" + self.get_mean_and_all_performance_string(self.total_precisions) + "\n"
        return performance_info

    @staticmethod
    def get_mean_and_all_performance_string(result_list):
        mean_value = sum(result_list) / len(result_list)
        res_str = "mean(" + str(mean_value) + "),"
        res_str += "all(" + "\t".join(["%.2f" % result for result in result_list]) + ")"
        return res_str

    def show_performance(self):
        print self.get_performance_info()
