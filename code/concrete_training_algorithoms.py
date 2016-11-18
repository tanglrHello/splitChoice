from nltk.classify import maxent
from abc import ABCMeta, abstractmethod
import random
import math


class BaseClassifier:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def train(self, train_dataset):
        pass

    @abstractmethod
    def batch_test(self, test_dataset):
        pass

    @abstractmethod
    # the returned value is such as 'y', indicating this data is predicted as 'y' by this classifier
    def single_test(self, test_data):
        pass

    @abstractmethod
    def batch_test_with_prob(self, test_dataset):
        pass

    @abstractmethod
    # the returned value is such as ('y', 0.4), while 0.4 indicates the probability for this predict
    def single_test_with_prob(self, test_data):
        pass


class MaxentClassifer(BaseClassifier):
    def __init__(self):
        self.classifier = None

    def train(self, train_dataset):
        encoding = maxent.TypedMaxentFeatureEncoding.train(train_dataset, count_cutoff=3, alwayson_features=True)
        self.classifier = maxent.MaxentClassifier.train(train_dataset, bernoulli=False, encoding=encoding, trace=0)
        # self.classifier.show_most_informative_features(10)
        return self

    def batch_test_with_prob(self, test_dataset):
        predict_result = []
        for test_data in test_dataset:
            predict_result.append(self.single_test_with_prob(test_data))
        return predict_result

    def single_test_with_prob(self, test_data):
        feature_vec = test_data[0]
        prob_dict = self.classifier.prob_classify(feature_vec)

        predict_res = prob_dict.max()
        predict_res_prob = prob_dict.prob(predict_res)

        return predict_res, predict_res_prob

    def batch_test(self, test_dataset):
        predict_result = []
        for test_data in test_dataset:
            predict_result.append(self.single_test(test_data))
        return predict_result

    def single_test(self, test_data):
        feature_vec = test_data[0]
        prob_dict = self.classifier.prob_classify(feature_vec)
        predict_res = prob_dict.max()
        return predict_res


class AdaboostClassifer(BaseClassifier):
    class WeightedClassifer:
        def __init__(self, classifier, weight):
            self.classifier = classifier
            self.weight = weight

    class WeightedTrainingData:
        def __init__(self, train_data, weight):
            self.train_data = train_data
            self.weight = weight

    def __init__(self, base_classifier, iterate_num, train_sample_prop=1):
        self.base_classifier = base_classifier
        self.weighted_classifiers = []
        self.iterate_num = iterate_num
        self.train_sample_prop = train_sample_prop

    def train(self, train_dataset):
        initial_train_data_weight = 1.0 / len(train_dataset)
        weighted_train_dataset = [self.WeightedTrainingData(data, initial_train_data_weight) for data in train_dataset]

        for i in range(self.iterate_num):
            train_dataset_sample = self.train_dataset_sampling(weighted_train_dataset)
            base_classfier = self.base_classifier.train(train_dataset_sample)
            classifier_weight = self.calculate_classifier_weight(base_classfier, train_dataset)
            self.weighted_classifiers.append(self.WeightedClassifer(base_classfier, classifier_weight))

            print "iterate index:" + str(i), "weight:" + str(classifier_weight)

            self.update_weights_for_training_dataset(base_classfier, weighted_train_dataset, classifier_weight)

        return self

    def batch_test(self, test_dataset):
        predict_results = []
        for test_data in test_dataset:
            predict_results.append(self.single_test(test_data))
        return predict_results

    def single_test(self, test_data):
        y_type_prob = 0
        for weighted_base_classifier in self.weighted_classifiers:
            res = weighted_base_classifier.classifier.single_test(test_data)
            classifier_weight = weighted_base_classifier.weight
            if res == 'y':
                y_type_prob += classifier_weight
            else:
                y_type_prob -= classifier_weight

        if y_type_prob > 0:
            return 'y'
        else:
            return 'n'

    def batch_test_with_prob(self, test_dataset):
        predict_results = self.batch_test(test_dataset)
        return [(res, -1) for res in predict_results]

    def single_test_with_prob(self, test_data):
        predict_result = self.single_test(test_data)
        return predict_result, -1

    @staticmethod
    def calculate_classifier_weight(base_classifier, train_dataset):
        error_num = 0

        predict_res = base_classifier.batch_test(train_dataset)
        for i, data in enumerate(train_dataset):
            if predict_res[i] != data[1]:
                error_num += 1

        error_rate = error_num/float(len(train_dataset))
        classifier_weight = 0.5 * math.log((1 - error_rate)/error_rate, math.e)

        return classifier_weight

    def update_weights_for_training_dataset(self, base_classifier, weighted_train_dataset, classifier_weight):
        predict_res = base_classifier.batch_test([data.train_data for data in weighted_train_dataset])

        multiplier = math.e**classifier_weight

        for index, res in enumerate(predict_res):
            if res == weighted_train_dataset[index].train_data[1]:
                new_weight = weighted_train_dataset[index].weight / multiplier
            else:
                new_weight = weighted_train_dataset[index].weight * multiplier
            weighted_train_dataset[index].weight = new_weight

        self.normalize_data_weights(weighted_train_dataset)
        return

    @staticmethod
    def normalize_data_weights(weighted_train_dataset):
        total_weights = 0
        for data in weighted_train_dataset:
            total_weights += data.weight

        for data in weighted_train_dataset:
            data.weight /= total_weights
        return

    def train_dataset_sampling(self, weighted_train_dataset):
        train_dataset_sample = []

        if len(weighted_train_dataset) == 0:
            raise Exception("no training data for sampling")

        data_weight_start_positions = [0]
        for weighted_train_data in weighted_train_dataset[1:]:
            data_weight_start_positions.append(data_weight_start_positions[-1] + weighted_train_data.weight)

        sample_train_data_num = self.train_sample_prop*len(weighted_train_dataset)
        for _ in range(sample_train_data_num):
            sample_position = random.random()
            index = self.find_sampled_data_index_in_dataset(data_weight_start_positions, sample_position)

            train_dataset_sample.append(weighted_train_dataset[index].train_data)

        return train_dataset_sample

    @staticmethod
    def find_sampled_data_index_in_dataset(data_weight_start_positions, sample_position):
        sorted_positions = sorted(data_weight_start_positions)
        if data_weight_start_positions != sorted_positions:
            raise Exception("data_weight_start_positions should be sorted")
        if data_weight_start_positions[0] != 0 or data_weight_start_positions[-1] >= 1:
            raise Exception("data_weight_start_positions should be within the range of 0 to 1")

        left = 0
        right = len(data_weight_start_positions)-1

        if data_weight_start_positions[left] == sample_position:
            return left
        if data_weight_start_positions[right] <= sample_position:
            return right

        if sample_position < data_weight_start_positions[left] or sample_position > 1:
            raise Exception("sample_position overflows the range of data_weight_start_positions")

        while True:
            medium = (left + right) / 2
            if sample_position < data_weight_start_positions[medium]:
                right = medium
            elif sample_position == data_weight_start_positions[medium]:
                return medium
            else:
                left = medium

            if left + 1 == right:
                return left
