# coding=utf-8
import requests

def test():
    segged_sentence = u"英国 比 美国\t气温 高 ， 日较差 低 。"
    pos = "NR VV NR NN AD PU NN AD PU"
    ner = "loc o loc o o o o o o"
    parameter = {"segged_sentence": segged_sentence, "pos": pos, "ner": ner}

    r = requests.post("http://172.28.61.247:8080/simple_split.html", data=parameter)
    print r.text


if __name__ == "__main__":
    test()