# coding=utf-8

FEATURE_NAMES = ['wordNumDiff',  # 拆分成的两部分各自词数差的绝对值
                 'charNumDiff',  # 拆分成的两部分各自字数差的绝对值
                 'postagEditDistance',  # 拆分成的两部分的词性序列的编辑距离
                 'lastPosComb',  # 拆分后两部分的最后一个词的词性组合
                 'lastPosEqual',  # 拆分后两部分的最后一个词的词性是否相同
                 'firstPosComb',  # 拆分后两部分的第一个词的词性组合
                 'firstPosEqual',  # 拆分后两部分的第一个词的词性是否相同
                 'lastWordInTimian',  # 题面中的最后一个词
                 'lastTwoWordsInTimian',  # 题面中的最后两个词拼接起来（如果只有一个词，前一个词用NULL）
                 'lastPostagInTimian',  # 题面中的最后一个词的词性
                 'timeCombination',  # 两个子句是否包含时间词的布尔值组合
                 'firstWordInSecondPart',  # 拆分后第二个部份的第一个词
                 # 'firstPostagInSecondPart',
                 # 'lastWordInFirstPart',
                 # 'containCuewordsComb',
                 "containCuewordsMain",  # 是否包含主要线索词
                 'bothContainLonLat'  # 拆分后两部分是否都包含经纬度
                 ]


class FeatureExtractor:
    def __init__(self, cueword_dict):
        self.cdict = cueword_dict

    @staticmethod
    def word_num_diff(seg_parts):
        # 当前假设只有一个逗号
        if len(seg_parts) != 2:
            raise Exception(u"应该只有两个部分：" + str(seg_parts))
        return abs(len(seg_parts[0]) - len(seg_parts[1]))

    @staticmethod
    def char_num_diff(text_parts):
        # 当前假设只有一个逗号
        if len(text_parts) != 2:
            raise Exception(u"应该只有两个部分：" + str(text_parts))
        return abs(len(text_parts[0]) - len(text_parts[1]))

    def edit_distance_of_postag(self, postag_parts):
        return self.edit_distance(postag_parts[0], postag_parts[1])

    @staticmethod
    def edit_distance(arr1, arr2):
        dis = [[0 for _ in range(len(arr1) + 1)] for _ in range(len(arr2) + 1)]

        for i in range(len(arr1) + 1):
            dis[0][i] = i
        for j in range(len(arr2) + 1):
            dis[j][0] = j

        for j in range(1, len(arr2) + 1):
            for i in range(1, len(arr1) + 1):
                if arr1[i - 1] == arr2[j - 1]:
                    dis[j][i] = dis[j - 1][i - 1]
                else:
                    dis[j][i] = min(dis[j - 1][i] + 1, dis[j][i - 1] + 1, dis[j - 1][i - 1] + 1)

        return dis[-1][-1]

    def pos_comb(self, ctype, postag_parts):
        return self.get_pos_comb(ctype, postag_parts[0], postag_parts[1])

    # 返回指定类型的词性组合，以及这对词性是否相同的布尔值；PosComb的功能函数
    @staticmethod
    def get_pos_comb(ctype, pos1, pos2):
        if len(pos1) == 0 or len(pos2) == 0:
            raise Exception(u"某个部分词性列表为空:" + str(pos1) + "--" + str(pos2))
        if ctype == "first":
            return pos1[0] + "/" + pos2[0], pos1[0] == pos2[0]
        elif ctype == "last":
            return pos1[-1] + "/" + pos2[-1], pos1[-1] == pos2[-1]

    @staticmethod
    def last_words_in_timian(tm_seg, wnum):
        if len(tm_seg) == 0:
            raise Exception("题面没有词")

        if wnum <= len(tm_seg):
            return "/".join(tm_seg[-wnum:])
        else:
            return "NULL/" * (wnum - len(tm_seg)) + "/".join(tm_seg)

    @staticmethod
    def time_in_each_part_comb(tm_seg, seg_parts, time_index_str):   # 根据新的时间标注规范修改!!!!!
        # 当前假设只有一个逗号
        if len(seg_parts) != 2:
            raise Exception(u"应该只有两个部分：" + str(seg_parts))

        part1_has_time = False
        part2_has_time = False

        first_xuanxiang_word_pos_in_setences = len(tm_seg)
        first_part2_word_pos_in_sentence = first_xuanxiang_word_pos_in_setences + len(seg_parts[0])
        total_word_num = first_part2_word_pos_in_sentence + len(seg_parts[1])

        # *(*%^&$^%$*%(^*&^%$%^&*^^$%#$%^&*()(*&^%$#@!@#$%^&*()!@#$%^&*()@#$%^&*()_@#$%^&*()
        time_index_list = [int(index) for index in time_index_str.split()]
        for time_index in time_index_list:
            '''
            print ti
            print txSplitPos,part2StartPos
            print part2StartPos,totalLen
            print txSplitPos<=ti<part2StartPos
            print part2StartPos<=ti<totalLen
            '''
            if first_xuanxiang_word_pos_in_setences <= time_index < first_part2_word_pos_in_sentence:
                part1_has_time = True
            elif first_part2_word_pos_in_sentence <= time_index < total_word_num:
                part2_has_time = True

        return str(part1_has_time) + "/" + str(part2_has_time)

    def contain_cuewords(self, seg_parts, cueword_feature_type):
        # 当前假设只有一个逗号
        if len(seg_parts) != 2:
            raise Exception(u"应该只有两个部分：" + str(seg_parts))

        cuewords_in_parts = [[], []]

        for index, seg in enumerate(seg_parts):
            for template_name in self.cdict:
                for cuewords in self.cdict[template_name]:
                    cuewords = cuewords.split("/")
                    flag = True
                    for cueword in cuewords:
                        if cueword not in seg:
                            flag = False

                    if flag:
                        cuewords_in_parts[index].append(template_name)
                        break

        cuewords_in_part1 = "None"
        cuewords_in_part2 = "None"
        if len(cuewords_in_parts[0]) > 0:
            cuewords_in_part1 = cuewords_in_parts[0][0]
        if len(cuewords_in_parts[1]) > 0:
            cuewords_in_part2 = cuewords_in_parts[1][0]

        if cueword_feature_type == "comb":
            return cuewords_in_part1 + "/" + cuewords_in_part2

        elif cueword_feature_type == "main":
            if u"影响" in cuewords_in_parts[0] or u"影响" in cuewords_in_parts[1]:
                return u"影响"
            elif u"因果" in cuewords_in_parts[0] or u"因果" in cuewords_in_parts[1]:
                return u"因果"
            elif u"条件" in cuewords_in_parts[0] or u'条件' in cuewords_in_parts[1]:
                return u'条件'
            elif u"趋势" in cuewords_in_parts[0] or u'趋势' in cuewords_in_parts[1]:
                return u'趋势'

    @staticmethod
    def both_contain_lonlat(seg_parts):
        text1 = "".join(seg_parts[0])
        text2 = "".join(seg_parts[1])
        if u"°" in text1 and u"°" in text2:
            flag1 = False
            flag2 = False
            for c in ['E', 'W', 'N', 'S']:
                if c in text1:
                    flag1 = True
                if c in text2:
                    flag2 = True
            return flag1 and flag2

        return False
