# -*- coding: utf-8 -*-
import os
from common import *


def get_cuewords(merged_data_path, cw_file_path):
    print_function_info("get_cuewords")
    cueword_dict = {}

    if os.path.exists(cw_file_path):
        load_existing_cueword_file(cw_file_path, cueword_dict)
    else:
        extract_cuewords_from_data(merged_data_path, cw_file_path, cueword_dict)

    return cueword_dict


def load_existing_cueword_file(cw_file_path, cueword_dict):
    print_function_info("load_existing_cueword_file")

    cw_file = open(cw_file_path)

    current_template_name = cw_file.readline().decode("utf-8").strip()
    cueword_dict[current_template_name] = {}
    for l in cw_file.readlines():
        l = l.decode('utf-8').strip()

        if l == "":   # indicate the end for one template
            current_template_name = ""
        if current_template_name == "":  # indicate the start for another template
            current_template_name = l
            cueword_dict[current_template_name] = {}
        else:   # read into cueword infos for current template
            l = l.split(":")
            cueword_dict[current_template_name][l[0]] = int(l[1])
    cw_file.close()


def extract_cuewords_from_data(merged_data_path, cw_file_path, cueword_dict):
    print_function_info("extract_cuewords_from_data")

    infile = open(merged_data_path)

    seg_index = 3

    top_template_index = 18
    top_cueword_index = 20
    second_template_index = 21
    second_cueword_index = 24

    infile.readline()
    for l in infile.readlines():
        l = l.decode("utf-8").strip().split(file_splitter)

        seg = l[seg_index].split()

        top_cueword = l[top_cueword_index].split()
        top_template = l[top_template_index]
        second_cueword = l[second_cueword_index].split()
        second_template = l[second_template_index]

        extract_template_cueword_pair(seg, top_cueword, top_template, cueword_dict)
        extract_template_cueword_pair(seg, second_cueword, second_template, cueword_dict)

    save_cueword(cueword_dict, cw_file_path)

    return cueword_dict


def extract_template_cueword_pair(seg, cueword_list, template_content, cueword_dict):
    for temlate_cueword in cueword_list:
        temlate_cueword = temlate_cueword.split("_")
        if len(temlate_cueword) > 2:
            raise Exception(u"多余的_")

        template_index = temlate_cueword[0]
        cueword_index = temlate_cueword[1]

        template_name = find_template_name(seg, template_index, template_content)
        curword_combination = get_cueword_combination(cueword_index, seg)

        if template_name not in cueword_dict:
            cueword_dict[template_name] = {}
        cueword_dict[template_name][curword_combination] = cueword_dict[template_name].get(curword_combination, 0) + 1


def find_template_name(seg, template_index, template_content):
    if template_content.count("_" + template_index) > 1:
        raise Exception(u"重复的模板下标:" + template_content)

    index_pos = template_content.find("_" + template_index)
    if index_pos == -1:
        raise Exception("".join(seg) + "; " + template_content)
    else:
        # 找到左边界
        left_index0 = -1
        if ',' not in template_content:   # for old template style
            left_index1 = template_content[:index_pos].rfind(u"）")
            left_index2 = template_content[:index_pos].rfind(u"（")   # deal with recursive template (may be deprecated)
            left_index3 = template_content[:index_pos].rfind(u"，")
        else:   # for new template style
            left_index1 = template_content[:index_pos].rfind(")")
            left_index2 = template_content[:index_pos].rfind("(")
            left_index3 = template_content[:index_pos].rfind(",")
        left_index4 = template_content[:index_pos].rfind(" ")

        left_index = max(left_index0, left_index1, left_index2, left_index3, left_index4)
        template_name = template_content[left_index + 1:index_pos].strip()

    return template_name


def get_cueword_combination(cueword_index, seg):
    cueword_index = cueword_index.split("-")
    this_template_cueword = []
    for i in cueword_index:
        this_template_cueword.append(seg[int(i)])
    cueword_combination = "/".join(this_template_cueword)

    return cueword_combination


def save_cueword(cueword_dict, cw_file_path):
    cueword_file = open(cw_file_path, "w")

    for template_name in cueword_dict:
        cueword_file.write(template_name.encode("utf-8") + "\n")
        for cueword in cueword_dict[template_name]:
            cueword_file.write(cueword.encode("utf-8") + ":" + str(cueword_dict[template_name][cueword]) + "\n")
        cueword_file.write("\n")

    cueword_file.close()


# cuewordFile.txt删掉了时间限定和实体信息陈述两类模板的线索词
# cuewordFile-sim.txt只保留影响、趋势、因果三类（有提升）
def get_simplified_cuewords(simplified_cueword_file_path):
    cueword_dict = {}
    load_existing_cueword_file(simplified_cueword_file_path, cueword_dict)
    return cueword_dict
