# -*- coding: utf-8 -*-
import os
from common import *


def merge_all_papers(ori_data_dir, merged_file_path):
    print_function_info("merge_all_papers")

    file_names = os.listdir(ori_data_dir)
    outfile = open(merged_file_path, "w")

    title_flag = False

    for f_name in file_names:
        if f_name.startswith("."):
            continue

        infile = open(ori_data_dir+"/"+f_name)
        title = infile.readline()
        if not title_flag:
            outfile.write("试卷名,"+title)
            title_flag = True

        for line in infile.readlines():
            outfile.write(f_name.split(file_splitter)[0]+file_splitter+line)

        infile.close()

    outfile.close()


def filter_split_data(merged_file_path, filtered_file_path):
    print_function_info("filter_split_data")

    infile = open(merged_file_path)
    outfile = open(filtered_file_path, "w")

    # 拆分信息在原文件表格中的列下标
    split_col_index = 3

    all_data_row_num = 0
    split_related_data_row_num = 0

    y_num = 0
    n_num = 0
    none_num = 0

    ori_y_num = 0
    mod_y_num = 0

    title = infile.readline()
    outfile.write(title)

    last_choice_split_tags = set()

    def count_mod_ori(tags):
        if "ed-mod" in tags:
            return 1, 0
        elif "ed-ori" in tags:
            return 0, 1
        else:
            return 0, 0

    for line in infile.readlines():
        all_data_row_num += 1
        try:
            tag = line.split(file_splitter)[split_col_index]
        except:
            print line
            print len(line.split(file_splitter))
            print split_col_index

        if tag != "None":
            outfile.write(line)
            split_related_data_row_num += 1
            if tag == "y":
                y_num += 1
            elif tag == "n":
                n_num += 1
            else:
                last_choice_split_tags.add(tag)
        else:
            none_num += 1

        if tag == "None" or tag == "y" or tag == "n":
            mod, ori = count_mod_ori(last_choice_split_tags)
            mod_y_num += mod
            ori_y_num += ori
            last_choice_split_tags.clear()

    mod, ori = count_mod_ori(last_choice_split_tags)
    mod_y_num += mod
    ori_y_num += ori

    infile.close()
    outfile.close()

    print "*****以下统计考虑所有拆分前、拆分后组合的试题文本*****"
    print "所有试题数据的行数\t" + str(all_data_row_num)
    print "含拆分信息的数据行数\t" + str(split_related_data_row_num)
    if all_data_row_num != 0:
        print "含拆分信息的数据所占比重\t" + str(split_related_data_row_num / float(all_data_row_num))
    print "-"*20

    total_ori_choice_num = none_num + y_num + n_num
    print "*****以下统计考虑所有拆分前的原始试题文本*****"
    print "共有原始选项个数：\t", total_ori_choice_num
    if total_ori_choice_num != 0:
        print "不含逗号的选项及其所占比例:\t", none_num, float(none_num) / total_ori_choice_num
        print "含逗号但不需拆分的选项及其比例：\t", n_num, float(n_num) / total_ori_choice_num
        print "含逗号且需要拆分的选项及其比例：\t", y_num, float(y_num) / total_ori_choice_num
    if y_num != 0:
        print y_num, ori_y_num, mod_y_num
        print "原始拆分(不需要补成分)的选项及其在拆分选项中的比例:\t", ori_y_num, float(ori_y_num) / y_num
        print "修改拆分(需要补充成分)的选项及其在拆分选项中的比例:\t", mod_y_num, float(mod_y_num) / y_num
