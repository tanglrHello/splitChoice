# coding=utf-8
import requests

def test():
    segged_sentence = u"上海 比 北京\t去年 气温 高 ， 降水量 大 。"
    pos = "NR V NR NT NN AV PU NN AV PU"
    ner = "loc O loc time term O O term O O"
    parameter = {"segged_sentence": segged_sentence, "pos": pos, "ner": ner}

    r = requests.post("http://localhost:8080/simple_split.html", data=parameter)
    print r.text


if __name__ == "__main__":
    test()