from pre_process import *
from get_cuewords import get_cuewords
from get_train_test_data import get_dataset_for_classifier
from train_test import TrainAndTest


def main():
    data_path = "../data/"

    # you should make sure all data in it has been tagged split/time/template info (attention!!!)
    # artificial postag tagging be empty is allowed, but then auto_pos is needed, or the data will be omitted
    ori_data_pathname = "11-5"

    ori_data_dir = data_path + ori_data_pathname
    merged_file_path = data_path + "merged_" + ori_data_pathname + ".data"
    merge_all_papers(ori_data_dir, merged_file_path)

    filtered_file_path = data_path + "filtered_" + ori_data_pathname + ".data"
    filter_split_data(merged_file_path, filtered_file_path)

    cueword_file_path = data_path + "cueword_"+ori_data_pathname + ".txt"
    cueword_dict = get_cuewords(filtered_file_path, cueword_file_path)

    classify_data_file_path = data_path + "classify_data_" + ori_data_pathname + ".data"
    all_dataset = get_dataset_for_classifier(cueword_dict,
                                             filtered_file_path,
                                             classify_data_file_path,
                                             force_generate_flag=True)

    test_prop = 0.1
    foldnum = 10
    y_prop_in_trainset = False
    predict_files_dir_path = data_path + "predict_result/"
    record_file_path = data_path + "auto_records.txt"
    my_classifier = TrainAndTest(all_dataset, test_prop, foldnum,
                                 predict_files_dir_path, record_file_path,
                                 y_prop_in_trainset)
    my_classifier.train_and_test()


if __name__ == "__main__":
    main()
