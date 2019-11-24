import math
from collections import OrderedDict


class Weighting:
    def __init__(self):
        self.documents = []
        self.features = []
        self.tf = [[]]
        self.idf = []
        self.tf_idf = [[]]
        self.mustReload = False

    def setText(self, source):
        self.documents = source
        self.features = list(OrderedDict((word, None) for document in self.documents for word in document).keys())
        self.tf = [[document.count(feature) for document in self.documents] for feature in self.getFeatures()]
        self.idf = [math.log10(len(termTfs) / sum(1 for tf in termTfs if tf > 0)) for termTfs in self.getTf()]
        self.tf_idf = [
            [(1 + math.log10(tf)) * idf if tf > 0 else tf for tf in termTfs]
            for termTfs, idf in zip(self.getTf(), self.getIdf())
        ]

    def getFeatures(self):
        return self.features

    def getTf(self):
        return self.tf

    def getIdf(self):
        return self.idf

    def getTfIdf(self):
        return self.tf_idf

    @staticmethod
    def normalisasi(weighting_2d_array):
        transposed_weighting_2d_array = [
            [weighting_2d_array[j][i] for j in range(len(weighting_2d_array))]
            for i in range(len(weighting_2d_array[0]))
        ]

        for i, row in enumerate(transposed_weighting_2d_array):
            divider = math.sqrt(sum([math.pow(element, 2) for element in row]))
            transposed_weighting_2d_array[i] = [element / divider for element in row]

        return [
            [transposed_weighting_2d_array[j][i] for j in range(len(transposed_weighting_2d_array))]
            for i in range(len(transposed_weighting_2d_array[0]))
        ]
