from pre_process import *
from get_cuewords import get_cuewords
from get_train_test_data import get_dataset_for_classifier
from train_test import train_and_test


def main():
    data_path = "../data/"
    ori_data_pathname = "11-5"

    ori_data_dir = data_path + ori_data_pathname
    merged_file_path = data_path + "merged_" + ori_data_pathname + ".data"
    merge_all_papers(ori_data_dir, merged_file_path)

    '''
    cueword_file_path = data_path + "cueword_"+ori_data_pathname + ".txt"
    cueword_dict = get_cuewords(merged_file_path, cueword_file_path)

    classify_data_file_path = data_path + "classify_data_" + ori_data_pathname + ".data"
    all_dataset = get_dataset_for_classifier(cueword_dict,
                                             merged_file_path,
                                             classify_data_file_path,
                                             force_generate_flag=True)

    test_prop = 0.1
    foldnum = 10
    predict_result_file_path = data_path + "predict_result.csv"
    test_record_file_path = data_path + "train&test_res.txt"
    train_and_test(all_dataset, test_prop, foldnum, predict_result_file_path, test_record_file_path)
    '''

if __name__ == "__main__":
    main()
