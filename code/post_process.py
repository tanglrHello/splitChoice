# coding=utf-8


def post_process(test_data, test_text, predict_result):
    post_predict_result = []

    for data, text, result in zip(test_data, test_text, predict_result):
        post_to_n_funcs1 = [specific_first_word_in_second_part,
                           specific_last_word_in_first_part,
                           both_contain_lonlat]
        post_to_n_funcs2 = [specific_word_in_timian,
                            specific_word_in_two_parts,
                            specific_word_in_first_part,
                            specific_word_in_second_part,]

        feature_vec = data[0]

        to_n_flag = False
        for func in post_to_n_funcs1:
            if func(feature_vec) == "n":
                post_predict_result.append("n")
                to_n_flag = True
                break

        if to_n_flag:
            continue

        for func in post_to_n_funcs2:
            if func(text) == "n":
                post_predict_result.append("n")
                to_n_flag = True
                break

        if not to_n_flag:
            post_predict_result.append(result)

    return post_predict_result


def specific_last_word_in_first_part(data):
    word_list = [u"时"]
    for word in word_list:
        if data['lastCharInFirstPart'] == word:
            return "n"


def specific_first_word_in_second_part(data):
    word_list = [u"利于", u"甚至", u"则", u"因此", u"便于", u"表示", u"但是", u"但", u"使", u"导致"]
    for word in word_list:
        if data['firstWordInSecondPart'] == word:
            return "n"


def both_contain_lonlat(data):
    if data['bothContainLonLat'] == True:
        return "n"


def specific_word_in_timian(text):
    word_list = [u"分别", u"及"]
    timian = text.split("\t")[0]
    for word in word_list:
        if word in timian:
            return "n"


def specific_word_in_two_parts(text):
    word_list = [(u"越", u"越")]

    xuanxiang = text.split("\t")[1].split(u"，")
    part1 = xuanxiang[0]
    part2 = xuanxiang[1]

    for word1, word2 in word_list:
        if word1 in part1 and word2 in part2:
            return "n"


def specific_word_in_first_part(text):
    word_list = [u"因为", u"由于", u"因"]
    xuanxiang = text.split("\t")[1]
    part1 = xuanxiang.split(u"，")[0]
    for word in word_list:
        if word in part1:
            return "n"


def specific_word_in_second_part(text):
    word_list = [u"使", u"导致"]
    xuanxiang = text.split("\t")[1]
    part2 = xuanxiang.split(u"，")[1]
    for word in word_list:
        if word in part2:
            return "n"