from pre_process import *
from get_cuewords import get_cuewords
from get_train_test_data import get_dataset_for_classifier, get_dataset_for_boundary
from train_test import TrainAndTest
# from draw_result import *
from concrete_training_algorithoms import *


def main():
    data_path = "../data/"
    ori_data_pathname = "11-5"
    y_prop_in_trainset = 0.4

    '''
    results = []
    y_props = []

    for _ in range(7):
        classify_data_file_path = data_path + "classify_data_" + ori_data_pathname + ".csv"
        all_dataset = get_dataset_for_classifier(cueword_dict,
                                                 filtered_file_path,
                                                 classify_data_file_path,
                                                 force_generate_flag=True)

        test_prop = 0.1
        foldnum = 10
        predict_files_dir_path = data_path + "predict_result/"
        record_file_path = data_path + "auto_records.txt"
        my_classifier = TrainAndTest(all_dataset, test_prop, foldnum,
                                     predict_files_dir_path, record_file_path,
                                     y_prop_in_trainset)
        my_classifier.train_and_test()

        results.append(my_classifier.get_mean_results_records())
        y_props.append(y_prop_in_trainset)

        if not y_prop_in_trainset:
            y_prop_in_trainset = 0.1
        else:
            y_prop_in_trainset += 0.1

    draw_result_for_features(results, y_props)
    '''
    cueword_dict = files_init(data_path, ori_data_pathname)
    filtered_file_path = data_path + "filtered_" + ori_data_pathname + ".data"
    classify_data_file_path = data_path + "classify_data_" + ori_data_pathname + ".csv"
    all_dataset = get_dataset_for_classifier(cueword_dict,
                                             filtered_file_path,
                                             classify_data_file_path,
                                             force_generate_flag=True)

    test_prop = 0.1
    foldnum = 10
    predict_files_dir_path = data_path + "predict_result/"
    record_file_path = data_path + "auto_records.txt"

    maxent_concrete_classifier = MaxentClassifer()
    # adaboost_concrete_classifier = AdaboostClassifer(maxent_concrete_classifier, 10)

    my_classifier = TrainAndTest(all_dataset, test_prop, foldnum,
                                 predict_files_dir_path, record_file_path,
                                 maxent_concrete_classifier,
                                 y_prop_in_trainset)
    my_classifier.train_and_test()


    # boundary_data_file_path = data_path + "boundary_data.csv"
    # get_dataset_for_boundary(filtered_file_path, boundary_data_file_path)


def files_init(data_path, ori_data_pathname):
    # you should make sure all data in it has been tagged split/time/template info (attention!!!)
    # artificial postag tagging be empty is allowed, but then auto_pos is needed, or the data will be omitted
    ori_data_dir = data_path + ori_data_pathname
    merged_file_path = data_path + "merged_" + ori_data_pathname + ".data"
    # merge_all_papers(ori_data_dir, merged_file_path)

    filtered_file_path = data_path + "filtered_" + ori_data_pathname + ".data"
    filter_split_data(merged_file_path, filtered_file_path)

    cueword_file_path = data_path + "cueword_" + ori_data_pathname + "_sim.txt"
    cueword_dict = get_cuewords(filtered_file_path, cueword_file_path)

    return cueword_dict



if __name__ == "__main__":
    main()
