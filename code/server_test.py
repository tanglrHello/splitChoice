# coding=utf-8
import requests

def test():

    segged_sentence = u"仅 考虑 地球运动 ， 图示 窗户 、 屋檐 的 搭建 对 室内 光热 的 影响 有	春分 到 夏至 ， 正午 屋檐 的 遮阳 作用 逐渐 增强 。"
    pos = "AD VV NN PU JJ NN PU NN DEG NN P NN NN DEG NN VE NN VV NN PU NN NN DEG NN NN AD VV PU"
    ner = "0 term 0 0 0 term 0 0 0 0 0 term 0 0 term 0 time time time 0 term 0 0 0 term term term 0"
    parameter = {"segged_sentence": segged_sentence, "pos": pos, "ner": ner}

    r = requests.post("http://10.0.2.11:8080/simple_split.html", data=parameter)
    print r.text

if __name__ == "__main__":
    test()