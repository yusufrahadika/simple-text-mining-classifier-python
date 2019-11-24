import numpy as np
from datetime import datetime
from Weighting import Weighting
from Preprocessing import Preprocessing
from collections import Counter, OrderedDict


class Klasifikasi:
    def __init__(self):
        self.weightingInstance = Weighting()
        self.fileClasses = []
        self.features = []
        self.uniqueClass = {}
        self.unique_class_list = []
        self.naive_bayes_model = [[]]

    def train(self, file_names, file_classes):
        print('train started =', datetime.now())
        self.weightingInstance.setText([
            Preprocessing.preprocess(open(file_name, 'r', encoding="ISO-8859-1").read())
            for file_name in file_names
        ])
        print('train feature started =', datetime.now())
        self.features = self.weightingInstance.getFeatures()
        feature_count = len(self.features)

        print('train class started =', datetime.now())
        self.fileClasses = file_classes
        self.uniqueClass = Counter(file_classes)
        self.unique_class_list = list(self.uniqueClass.keys())
        class_count = len(self.uniqueClass)

        # train naive-bayes
        naive_bayes_count = np.zeros((feature_count, class_count), dtype=int)
        for document_index, (class_name, document) in enumerate(zip(file_classes, self.weightingInstance.documents)):
            column_index = self.unique_class_list.index(class_name)
            for row_index, feature in enumerate(self.features):
                naive_bayes_count[row_index][column_index] += self.weightingInstance.tf[row_index][document_index]

        classes_word_count = np.zeros(class_count, dtype=int)

        for row_naive_bayes_count in naive_bayes_count:
            for column_index, count in enumerate(row_naive_bayes_count):
                classes_word_count[column_index] += count

        self.naive_bayes_model = [
            [((naive_bayes_count[feature_index, class_index] + 1) / (classes_word_count[class_index] + feature_count))
             for class_index in range(class_count)]
            for feature_index in range(feature_count)
        ]

    # 0 : Naive-bayes with Laplace smoothing
    # 1 : Rocchio
    def test(self, test_file_names, method_code=0):
        if method_code == 0:
            return self.naive_bayes(test_file_names)
        elif method_code == 1:
            return self.rocchio(test_file_names)
        else:
            raise Exception(f'Method code {method_code} is not valid')

    def naive_bayes(self, test_file_names):
        print('test started =', datetime.now())
        result = []

        total_train_document_count = len(self.fileClasses)
        initial_naive_bayes_probability = [
            sum(1 for class_name in self.fileClasses if class_name == unique_class) / total_train_document_count
            for unique_class in self.uniqueClass.keys()
        ]

        for i, test_file_name in enumerate(test_file_names):
            print('file', i + 1, 'classification started =', datetime.now())
            test_file_weighting_features = Preprocessing.type(
                Preprocessing.preprocess(open(test_file_name, 'r', encoding="ISO-8859-1").read())
            )

            naive_bayes_class_probability = initial_naive_bayes_probability.copy()
            for class_index, data_train_class in enumerate(self.unique_class_list):
                for test_file_weighting_feature in test_file_weighting_features:
                    if test_file_weighting_feature in self.features:
                        term_data_train_index = self.features.index(test_file_weighting_feature)

                        naive_bayes_class_probability[class_index] *= \
                            self.naive_bayes_model[term_data_train_index][class_index]

            print(naive_bayes_class_probability)
            result.\
                append(self.unique_class_list[naive_bayes_class_probability.index(max(naive_bayes_class_probability))])

        return result

    def rocchio(self, test_file_names):
        print('test started =', datetime.now())
        result = []

        data_train_classes = OrderedDict((file_class, None) for file_class in self.fileClasses)
        normalized_tf_idf = Weighting.normalisasi(self.weightingInstance.getTfIdf())

        for data_train_class in data_train_classes.keys():
            document_class_indexes = [
                document_class_index
                for document_class_index, file_class in enumerate(self.fileClasses)
                if file_class == data_train_class
            ]

            data_train_classes[data_train_class] = [
                np.mean([
                    weight
                    for column_index, weight in enumerate(row_weighting)
                    if column_index in document_class_indexes
                ])
                for feature, row_weighting in zip(self.weightingInstance.getFeatures(), normalized_tf_idf)
            ]

        for i, test_file_name in enumerate(test_file_names):
            print('file', i + 1, 'classification started =', datetime.now())

            query_term = Preprocessing.preprocess(open(test_file_name, 'r', encoding="ISO-8859-1").read())
            query_tf = [[query_term.count(feature)] for feature in self.weightingInstance.getFeatures()]

            normalized_query_tf_idf = Weighting.normalisasi([
                [query_term_freq[0] * idf_term]
                for query_term_freq, idf_term in zip(query_tf, self.weightingInstance.getIdf())
            ])

            rocchio_distance = [
                1 - sum([query_row[0] * class_feature_mean for query_row, class_feature_mean in
                         zip(normalized_query_tf_idf, data_train_class_item)])
                for _, data_train_class_item in data_train_classes.items()
            ]
            result.append(list(data_train_classes.keys())[rocchio_distance.index(min(rocchio_distance))])

        return result

    @staticmethod
    def hitungAkurasi(test_classes, actual_classes):
        count_true = \
            sum(1 for test_class, actual_class in zip(test_classes, actual_classes) if test_class == actual_class)

        return count_true / min(len(test_classes), len(actual_classes))
