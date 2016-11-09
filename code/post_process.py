# coding=utf-8


class PostProcessor:
    def __init__(self):
        # these funtions will use features in feature_vec of each data
        self.post_to_n_funcs1 = [self.specific_first_word_in_second_part,
                                 self.specific_last_word_in_first_part,
                                 self.both_contain_lonlat]

        # these functions will only use the original text of that choice
        self.post_to_n_funcs2 = [self.specific_word_in_timian,
                                 self.specific_word_in_two_parts,
                                 self.specific_word_in_first_part,
                                 self.specific_word_in_second_part]

        self.y_to_n_right = 0
        self.y_to_n_wrong = 0
        self.n_to_y_right = 0
        self.n_to_y_wrong = 0

    def post_process(self, test_set, test_text, predict_result):
        post_predict_result = []

        for test_single_data, text, result in zip(test_set, test_text, predict_result):
            to_n_flag = False
            for func in self.post_to_n_funcs1:
                if func(test_single_data.full_feature_vec) == "n":
                    post_predict_result.append("n")
                    to_n_flag = True
                    break

            if to_n_flag:
                continue

            for func in self.post_to_n_funcs2:
                if func(text) == "n":
                    post_predict_result.append("n")
                    to_n_flag = True
                    break

            if not to_n_flag:
                post_predict_result.append(result)

        self.add_effects(test_set, predict_result, post_predict_result)

        return post_predict_result

    def add_effects(self, test_set, predict_results, post_predict_results):
        real_labels = [single_data.data_for_train_test[1] for single_data in test_set]
        for real_label, predict_label, post_label in zip(real_labels, predict_results, post_predict_results):
            if predict_label != post_label:
                if predict_label == "y":
                    if post_label == real_label:
                        self.y_to_n_right += 1
                    else:
                        self.y_to_n_wrong += 1
                elif predict_label == "n":
                    if post_label == real_label:
                        self.n_to_y_right += 1
                    else:
                        self.n_to_y_wrong += 1

    def show_post_process_effect(self):
        print self.get_post_process_info()

    def get_post_process_info(self):
        info = ""
        info += "y_to_n_right:" + str(self.y_to_n_right) + "\n"
        info += "y_to_n_wrong:" + str(self.y_to_n_wrong) + "\n"
        info += "n_to_y_right:" + str(self.n_to_y_right) + "\n"
        info += "n_to_y_wrong:" + str(self.n_to_y_wrong) + "\n"
        return info

    @staticmethod
    def specific_last_word_in_first_part(data):
        word_list = [u"时"]
        for word in word_list:
            if data['lastCharInFirstPart'] == word:
                return "n"

    @staticmethod
    def specific_first_word_in_second_part(data):
        word_list = [u"利于", u"甚至", u"则", u"因此", u"便于", u"表示", u"但是", u"但", u"使", u"导致"]
        for word in word_list:
            if data['firstWordInSecondPart'] == word:
                return "n"

    @staticmethod
    def both_contain_lonlat(data):
        if data['bothContainLonLat']:
            return "n"

    @staticmethod
    def specific_word_in_timian(text):
        word_list = [u"分别", u"及"]
        timian = text.split("\t")[0]
        for word in word_list:
            if word in timian:
                return "n"

    @staticmethod
    def specific_word_in_two_parts(text):
        word_list = [(u"越", u"越")]

        xuanxiang = text.split("\t")[1].split(u"，")
        part1 = xuanxiang[0]
        part2 = xuanxiang[1]

        for word1, word2 in word_list:
            if word1 in part1 and word2 in part2:
                return "n"

    @staticmethod
    def specific_word_in_first_part(text):
        word_list = [u"因为", u"由于", u"因"]
        xuanxiang = text.split("\t")[1]
        part1 = xuanxiang.split(u"，")[0]
        for word in word_list:
            if word in part1:
                return "n"

    @staticmethod
    def specific_word_in_second_part(text):
        word_list = [u"使", u"导致"]
        xuanxiang = text.split("\t")[1]
        part2 = xuanxiang.split(u"，")[1]
        for word in word_list:
            if word in part2:
                return "n"
