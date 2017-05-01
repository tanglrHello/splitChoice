# coding=utf-8
import requests

def test():
    segged_sentence = u"图5 中 图例 甲 、 乙 的 岩石 类型 分别 是\t沉积岩 ， 变质岩 。"
    pos = "NN LC NN NN PU NN DEG NN NN AD VC NN PU NN PU"
    ner = "0 0 0 0 0 0 0 term term 0 0 term 0 term 0"
    parameter = {"segged_sentence": segged_sentence, "pos": pos, "ner": ner}

    r = requests.post("http://10.0.2.11:8080/simple_split.html", data=parameter)
    print r.text

if __name__ == "__main__":
    test()