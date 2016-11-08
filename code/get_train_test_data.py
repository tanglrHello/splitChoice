# coding=utf-8
import os
import random
from common import *
from feature_extractor import *


class SingleData:
    def __init__(self, full_feature_vec, filtered_feature_vec, label, ori_text):
        self.data_for_train_test = (filtered_feature_vec, label)
        self.full_feature_vec = full_feature_vec
        self.ori_text = ori_text


# -暂时只考虑拆分成两部分的句子
# -不划分训练集测试集，所有样本均返回，返回的样本中没有样本的来源信息
# -you should set force_generate_flag to False when you are not sure whether the existing classify_data_file is
# corresponding to the cueword_dict you pass in
def get_dataset_for_classifier(cueword_dict, merged_file_path, classify_data_file_path, force_generate_flag=False):
    print_function_info("get_dataset_for_classifier")

    if os.path.exists(classify_data_file_path) and not force_generate_flag:
        return load_dataset_for_classifier(classify_data_file_path)
    else:
        return generate_dataset_for_classifier(cueword_dict, merged_file_path, classify_data_file_path)


def load_dataset_for_classifier(classify_data_file_path):
    print_function_info("load_dataset_for_classifier")

    all_dataset = {"y": [], "n": []}

    classify_data_file = open(classify_data_file_path)
    titles = classify_data_file.readline().split(out_file_splitter)

    text_index_in_file = 1
    feature_start_index_in_file = 4
    label_index_in_file = -1

    for line in classify_data_file.readlines():
        line = line.split(out_file_splitter)

        full_feature_vector = {}
        for index, feature_value in enumerate(line[feature_start_index_in_file:-1]):
            full_feature_vector[titles[index + feature_start_index_in_file]] = feature_value

        ori_text = line[text_index_in_file]
        data_label = line[label_index_in_file]

        filtered_feature_vector = filter_features(full_feature_vector)
        single_data = SingleData(full_feature_vector, filtered_feature_vector, data_label, ori_text)

        all_dataset[data_label].append(single_data)

    return all_dataset


def generate_dataset_for_classifier(cueword_dict, filtered_file_path, classify_data_file_path):
    print_function_info("generate_dataset_for_classifier")

    related_data_num = 0
    many_parts_data_num = 0
    no_postag_data_num = 0

    infile = open(filtered_file_path)
    outfile = open(classify_data_file_path, "w")

    all_dataset = {"y": [], "n": []}

    write_file_title(outfile)

    for l in infile.readlines():
        infos = l.strip().decode("utf-8").split(file_splitter)

        split_info = infos[index_dict['splitinfo']]

        if split_info == "y" or split_info == "n":
            related_data_num += 1
            text_info = dict()

            for field_name in index_dict:
                text_info[field_name] = infos[index_dict[field_name]]

            is_valid = modify_data_without_postag_tagging(text_info)
            if not is_valid:
                no_postag_data_num += 1
                continue

            # ignore choices with more than two split parts
            if len(text_info['text'].split("\t")[1].split(u"，")) > 2:
                many_parts_data_num += 1
                continue

            try:
                single_data = get_featured_data_for_classify(cueword_dict, text_info)
            except Exception, e:
                try:
                    print "error:", e.message.encode("utf-8")
                except:
                    print "error:", e.message
                continue

            full_feature_vector = single_data.full_feature_vec
            data_label = single_data.data_for_train_test[1]

            # outfile record more data than the returned all_dataset
            outfile.write(text_info['source'].encode("utf-8") + file_splitter)
            outfile.write(text_info['text'].encode("utf-8") + file_splitter)
            outfile.write(text_info['segres'].encode("utf-8") + file_splitter)
            outfile.write(text_info['posres'].encode("utf-8"))

            for feature_name in full_feature_vector:
                try:
                    outfile.write(out_file_splitter + str(full_feature_vector[feature_name]).encode("utf-8"))
                except:
                    outfile.write(out_file_splitter + str(full_feature_vector[feature_name].encode("utf-8")))

            outfile.write(file_splitter + data_label.encode("utf-8") + "\n")

            all_dataset[data_label].append(single_data)
        else:
            continue

    infile.close()
    outfile.close()

    print "related data num:\t", related_data_num
    print "no postag data num:\t", no_postag_data_num
    print "many parts data num:\t", many_parts_data_num
    print "available data num:\t", related_data_num - no_postag_data_num - many_parts_data_num

    return all_dataset


def modify_data_without_postag_tagging(text_info):
    if text_info["posres"] == "":
        if text_info["auto_pos"] == "":
            print text_info['source']
            return False
        else:
            text_info['posres'] = text_info['auto_pos']
    return True


def write_file_title(outfile):
    outfile.write(file_splitter.join(['source', 'oritext', 'seg', 'postag']))
    for fvt in FEATURE_NAMES:
        outfile.write(out_file_splitter + fvt)
    outfile.write(out_file_splitter + "label\n")


def get_featured_data_for_classify(cueword_dict, text_info):
    label = text_info['splitinfo']

    timian_xuanxiang = text_info['text'].split("\t")
    timian_text = timian_xuanxiang[0]
    xuanxiang_text = timian_xuanxiang[1]

    seg = text_info['segres'].split()
    postag = text_info['posres'].split()

    for i in range(len(seg)):
        if "".join(seg[:i]) == timian_text:
            xuanxiang_start_index = i
            break
    else:
        raise Exception("invalid seg res: no space between timian and xuanxiang")

    timian_seg = seg[:xuanxiang_start_index]
    timian_postag = postag[:xuanxiang_start_index]
    xuanxiang_seg = seg[xuanxiang_start_index:]
    xuanxiang_postag = postag[xuanxiang_start_index:]

    text_parts = xuanxiang_text.split(u"，")

    if len(text_parts) != 2:
        raise Exception(u"应该有且只有一个逗号：" + xuanxiang_text)

    split_position = xuanxiang_seg.index(u"，")

    seg_parts = [xuanxiang_seg[:split_position], xuanxiang_seg[split_position + 1:]]
    postag_parts = [xuanxiang_postag[:split_position], xuanxiang_postag[split_position + 1:]]

    fe = FeatureExtractor(cueword_dict)
    feature_vector = dict()

    feature_vector['wordNumDiff'] = fe.word_num_diff(seg_parts)
    feature_vector['charNumDiff'] = fe.char_num_diff(text_parts)
    feature_vector['postagEditDistance'] = fe.edit_distance_of_postag(postag_parts)

    last_pos_pair = fe.pos_comb("last", postag_parts)
    feature_vector['lastPosComb'] = last_pos_pair[0]
    feature_vector['lastPosEqual'] = last_pos_pair[1]

    first_pos_pair = fe.pos_comb("first", postag_parts)
    feature_vector['firstPosComb'] = first_pos_pair[0]
    feature_vector['firstPosEqual'] = first_pos_pair[1]

    feature_vector['lastWordInTimian'] = fe.last_words_in_timian(timian_seg, 1)
    feature_vector['lastTwoWordsInTimian'] = fe.last_words_in_timian(timian_seg, 2)
    feature_vector['lastPostagInTimian'] = timian_postag[-1]

    feature_vector['timeCombination'] = fe.time_in_each_part_comb(timian_seg, seg_parts, text_info['goldtimes'])

    feature_vector['containCuewordsComb'] = fe.contain_cuewords(seg_parts, "comb")
    feature_vector['containCuewordsMain'] = fe.contain_cuewords(seg_parts, "main")

    feature_vector['firstWordInSecondPart'] = seg_parts[1][0]
    feature_vector['firstPostagInSecondPart'] = postag_parts[1][0]
    feature_vector['lastWordInFirstPart'] = seg_parts[0][-1]
    feature_vector['lastCharInFirstPart'] = seg_parts[0][-1][-1]

    feature_vector['bothContainLonLat'] = fe.both_contain_lonlat(seg_parts)

    filtered_feature_vector = filter_features(feature_vector)
    data = SingleData(feature_vector, filtered_feature_vector, label, text_info['text'])

    return data


def filter_features(full_feature_vec):
    filtered_feature_vec = {}

    for feature_name in FEATURE_NAMES:
        filtered_feature_vec[feature_name] = full_feature_vec[feature_name]

    return filtered_feature_vec


def get_dataset_for_boundary():
    pass


def construct_y_prop_dataset(ori_dataset, y_prop):
    print_function_info("construct_y_prop_dataset")
    print "y_prop="+str(y_prop)
    y_data = ori_dataset['y'][:]
    n_data = ori_dataset['n'][:]

    total_data_num = len(y_data) + len(n_data)

    balanced_dataset = dict()
    balanced_dataset['y'] = y_data
    balanced_dataset['n'] = n_data

    if len(y_data) / float(total_data_num) < y_prop:
        total_target_size = len(n_data) / (1 - y_prop)
        y_target_size = int(total_target_size - len(n_data))
        enlarge_dataset(y_data, y_target_size)
    else:
        total_target_size = len(y_data) / y_prop
        n_target_size = int(total_target_size - len(y_data))
        enlarge_dataset(n_data, n_target_size)

    return balanced_dataset


def enlarge_dataset(ori_datas, target_size):
    ori_size = len(ori_datas)
    for i in range(target_size - ori_size):
        selected_index = random.randint(0, ori_size-1)
        ori_datas.append(ori_datas[selected_index])


# test_prop将原始数据的多少拿出作为测试集，例如0.1表示将10%的数据作为测试集
# 训练集和测试集中，n和y两种类型的数据比例一致
def split_dataset(ori_dataset, test_prop, foldnum, fold_index, y_prop_in_trainset=False):
    print_function_info("split_dataset")

    n_data = ori_dataset['n']
    y_data = ori_dataset['y']

    foldlen_n = len(n_data) / foldnum
    foldlen_y = len(y_data) / foldnum

    first_test_pos_for_n = foldlen_n * (fold_index % foldnum)
    last_test_pos_for_n = foldlen_n * (fold_index % foldnum) + int(len(n_data) * test_prop)
    first_test_pos_for_y = foldlen_y * (fold_index % foldnum)
    last_test_pos_for_y = foldlen_y * (fold_index % foldnum) + int(len(y_data) * test_prop)

    train_set = dict()
    train_set['n'] = n_data[:first_test_pos_for_n] + n_data[last_test_pos_for_n:]
    train_set['y'] = y_data[:first_test_pos_for_y] + y_data[last_test_pos_for_y:]

    test_set = dict()
    test_set['n'] = n_data[first_test_pos_for_n:last_test_pos_for_n]
    test_set['y'] = y_data[first_test_pos_for_y:last_test_pos_for_y]

    '''
        print u"  所有y数据的个数:\t" + str(len(n_data))
        print u"  所有n数据的个数:\t" + str(len(y_data))
        print u"  训练集大小:\t" + str(len(train_set))
        print u"  测试集大小:\t" + str(len(test_set))
    '''

    print_dataset_info("test_dataset", test_set)
    if y_prop_in_trainset:
        modified_train_set = construct_y_prop_dataset(train_set, y_prop_in_trainset)
        print_dataset_info("balanced_train_dataset", modified_train_set)
        return modified_train_set['y'] + modified_train_set['n'], test_set['y'] + test_set['n']
    else:
        print_dataset_info("train_dataset", train_set)
        return train_set['y'] + train_set['n'], test_set['y'] + test_set['n']


def print_dataset_info(dataset_name, dataset):
    print "y in " + dataset_name + ":", len(dataset['y'])
    print "n in " + dataset_name + ":", len(dataset['n'])
