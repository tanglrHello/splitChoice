# coding=utf-8
from collections import Counter
from bottle import Bottle, route, run, request, response, get, post
from main import files_init
from get_train_test_data import SingleData, get_featured_data_for_classify
from get_train_test_data import get_dataset_for_classifier, construct_y_prop_dataset
from concrete_training_algorithoms import *

app = Bottle()
DATA_PATH = "../data/"
ORI_DATA_PATHNAME = "11-5"
CUEWORD_DICT = files_init(DATA_PATH, ORI_DATA_PATHNAME)
CLASSIFIER = None

@app.get("/test.html")
def test():
    return "hello"


@app.post("/testpost.html")
def testpost():
    data = request.forms
    print data.get("key","no_key")
    print data.get("v", "no_v")
    return data.get("key","no_key")


@app.post("/simple_split.html")
def simple_split():
    data = request.forms
    segged_sentence = data.get("segged_sentence", None).decode("utf-8")
    pos = data.get("pos", None).split()
    ner = data.get("ner", None).split()

    assert ner and segged_sentence and pos
    assert len(segged_sentence.split()) == len(pos) == len(ner)

    timian = segged_sentence.split('\t')[0].split()
    xuanxiang = segged_sentence.split('\t')[1].split()
    all_words = segged_sentence.split()

    assert u"，" in xuanxiang

    # predict for each pair of comma split part
    results = []
    comma_pos = [] # store the position of commas
    for index, word in enumerate(xuanxiang):
        if word == u"，":
            comma_pos.append(index + len(timian))
    part_boundaries = [len(timian) - 1] + comma_pos + [len(all_words)]

    if all_words[-1] == u"。":
        del all_words[-1]
        del ner[-1]
        del pos[-1]

    for i in range(len(part_boundaries) - 2):
        for j in range(i + 1, len(part_boundaries) - 1):
            tmp_xuanxiang = all_words[part_boundaries[i] + 1 : part_boundaries[i + 1]] + [u"，"]\
                        + all_words[part_boundaries[j] + 1 : part_boundaries[j + 1]] + [u"。"]
            tmp_pos = pos[:len(timian)] + pos[part_boundaries[i] + 1 : part_boundaries[i + 1]] + ["PU"]\
                        + pos[part_boundaries[j] + 1 : part_boundaries[j + 1]] + ["PU"]
            tmp_ner = ner[:len(timian)] + ner[part_boundaries[i] + 1 : part_boundaries[i + 1]] + ["O"]\
                        + ner[part_boundaries[j] + 1 : part_boundaries[j + 1]] + ["O"]

            text_info = dict()
            text_info['splitinfo'] = None
            text_info['text'] = "".join(timian) + "\t" + "".join(tmp_xuanxiang)
            text_info['segres'] = " ".join(timian + xuanxiang)
            text_info['posres'] = " ".join(tmp_pos)
            text_info['goldtimes'] = []
            for k, tag in enumerate(tmp_ner):
                if tag == "time":
                    text_info['goldtimes'].append(str(k))
            text_info['goldtimes'] = " ".join(text_info['goldtimes'])

            global CUEWORD_DICT
            single_data = get_featured_data_for_classify(CUEWORD_DICT, text_info)
            results.append(test(single_data))

    # generate final prediction
    statistic = Counter(results)
    return statistic.most_common(1)[0][0]

def train():
    global CUEWORD_DICT
    global CLASSIFIER
    global DATA_PATH
    global ORI_DATA_PATHNAME

    y_prop_in_trainset = 0.4

    classify_data_file_path = DATA_PATH + "classify_data_" + ORI_DATA_PATHNAME + ".csv"
    filtered_file_path = DATA_PATH + "filtered_" + ORI_DATA_PATHNAME + ".data"
    all_dataset = get_dataset_for_classifier(CUEWORD_DICT,
                                             filtered_file_path,
                                             classify_data_file_path,
                                             force_generate_flag=True)

    train_data_dict= construct_y_prop_dataset(all_dataset, y_prop_in_trainset)
    train_data = train_data_dict['n'] + train_data_dict['y']
    real_train_data = [single_data.data_for_train_test for single_data in train_data]

    CLASSIFIER = MaxentClassifer()
    CLASSIFIER.train(real_train_data)


def test(single_data):
    test_data = single_data.data_for_train_test
    global CLASSIFIER
    return CLASSIFIER.single_test(test_data)


if __name__ == "__main__":
    train()
    run(app, host='172.28.61.247', port=8080)
