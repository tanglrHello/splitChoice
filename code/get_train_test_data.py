# coding=utf-8
import os
from common import *
from feature_extractor import *


class SingleData:
    def __init__(self, data, ori_text):
        self.data = data
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

        feature_vector = {}
        for index, feature_value in enumerate(line[feature_start_index_in_file:-1]):
            feature_vector[titles[index + feature_start_index_in_file]] = feature_value

        ori_text = line[text_index_in_file]
        data_label = line[label_index_in_file]

        single_data = SingleData((feature_vector, data_label), ori_text)
        all_dataset[data_label].append(single_data)

    return all_dataset


def generate_dataset_for_classifier(cueword_dict, merged_file_path, classify_data_file_path):
    print_function_info("generate_dataset_for_classifier")

    infile = open(merged_file_path)
    outfile = open(classify_data_file_path, "w")

    all_dataset = {"y": [], "n": []}

    # 写输出文件的标题
    outfile.write(file_splitter.join(['source', 'oritext', 'seg', 'postag']))
    for fvt in FEATURE_NAMES:
        outfile.write(out_file_splitter + fvt)
    outfile.write(out_file_splitter + "label\n")

    # merged file has one more column for source compared with raw data file
    index_offset = 1

    for l in infile.readlines():
        infos = l.strip().decode("utf-8").split(file_splitter)

        split_info = infos[index_dict['splitinfo'] + index_offset]

        if split_info == "y" or split_info == "n":
            text_info = dict()
            text_info['source'] = infos[0]
            for field_name in index_dict:
                text_info[field_name] = infos[index_dict[field_name] + index_offset]

            # ignore choices with more than two split parts
            if len(text_info['text'].split("\t")[1].split(u"，")) > 2:
                continue

            try:
                feature_label_data = get_featured_data_for_classify(cueword_dict, text_info)
                feature_vec = feature_label_data[0]
                data_label = feature_label_data[1]
            except Exception, e:
                print e
                continue

            # outfile record more data than the returned all_dataset
            outfile.write(text_info['source'].encode("utf-8") + file_splitter)
            outfile.write(text_info['text'].encode("utf-8") + file_splitter)
            outfile.write(text_info['segres'].encode("utf-8") + file_splitter)
            outfile.write(text_info['posres'].encode("utf-8"))

            for feature_name in FEATURE_NAMES:
                try:
                    outfile.write(out_file_splitter + str(feature_vec[feature_name]).encode("utf-8"))
                except:
                    outfile.write(out_file_splitter + str(feature_vec[feature_name].encode("utf-8")))

            outfile.write(file_splitter + data_label.encode("utf-8") + "\n")

            single_data = SingleData(feature_label_data, text_info['text'])
            all_dataset[data_label].append(single_data)
        else:
            continue

    infile.close()
    outfile.close()

    return all_dataset


def get_featured_data_for_classify(cueword_dict, test_info):
    label = test_info['splitinfo']

    timian_xuanxiang = test_info['text'].split("\t")
    timian_text = timian_xuanxiang[0]
    xuanxiang_text = timian_xuanxiang[1]

    seg = test_info['segres'].split()
    postag = test_info['posres'].split()

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

    feature_vector['timeCombination'] = fe.time_in_each_part_comb(timian_seg, seg_parts, test_info['time'])
    feature_vector['containCuewordsComb'] = fe.contain_cuewords(seg_parts, "comb")
    feature_vector['containCuewordsMain'] = fe.contain_cuewords(seg_parts, "main")

    feature_vector['firstWordInSecondPart'] = seg_parts[1][0]
    feature_vector['firstPostagInSecondPart'] = postag_parts[1][0]
    feature_vector['lastWordInFirstPart'] = seg_parts[0][-1]

    feature_vector['bothContainLonLat'] = fe.both_contain_lonlat(seg_parts)

    deleted_features = []
    for feature_name in feature_vector:
        if feature_name not in FEATURE_NAMES:
            deleted_features.append(feature_name)

    for feature_name in deleted_features:
        del feature_vector[feature_name]

    data = (feature_vector, label)
    return data


def get_dataset_for_boundary():
    pass


# test_prop将原始数据的多少拿出作为测试集，例如0.1表示将10%的数据作为测试集
# 训练集和测试集中，n和y两种类型的数据比例一致
def split_dataset(ori_dataset, test_prop, foldnum, fold_index):
    print_function_info("split_dataset")

    n_data = ori_dataset['n']
    y_data = ori_dataset['y']

    foldlen_n = len(n_data) / foldnum
    foldlen_y = len(y_data) / foldnum

    first_test_pos_for_n = foldlen_n * (fold_index % foldnum)
    last_test_pos_for_n = foldlen_n * (fold_index % foldnum) + int(len(n_data) * test_prop)
    first_test_pos_for_y = foldlen_y * (fold_index % foldnum)
    last_test_pos_for_y = foldlen_y * (fold_index % foldnum) + int(len(y_data) * test_prop)

    train_set = n_data[:first_test_pos_for_n] + n_data[last_test_pos_for_n:] + \
                y_data[:first_test_pos_for_y] + y_data[last_test_pos_for_y:]
    test_set = n_data[first_test_pos_for_n:last_test_pos_for_n] + y_data[first_test_pos_for_y:last_test_pos_for_y]

    print u"  所有y数据的个数:\t" + str(len(n_data))
    print u"  所有n数据的个数:\t" + str(len(y_data))
    print u"  训练集大小:\t" + str(len(train_set))
    print u"  测试集大小:\t" + str(len(test_set))

    return train_set, test_set
