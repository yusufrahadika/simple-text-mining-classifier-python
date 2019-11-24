import re
import string
from collections import OrderedDict
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


class Preprocessing:

    punctuation_list = string.punctuation
    match_domain = re.compile(
        r'[\S]+\.(net|com|org|info|edu|gov|uk|de|ca|jp|fr|au|us|ru|ch|it|nel|se|no|es|mil)[\S]*\s?')
    stopwords = open("stopword-list.txt", "r").read().split("\n")

    @staticmethod
    def cleaning(source):
        source_to_process = source.strip()
        source_to_process = Preprocessing.match_domain.sub('', source_to_process)
        result = ''

        source_last_char_index = len(source_to_process) - 1
        for i, char in enumerate(source_to_process):
            if char in Preprocessing.punctuation_list or char.isnumeric():
                if i != source_last_char_index:
                    next_char = source_to_process[i + 1]
                    if (next_char not in Preprocessing.punctuation_list and
                            not next_char.isnumeric() and next_char != ' '):
                        result += ' '
            else:
                result += char

        return result

    @staticmethod
    def case_folding(source):
        return source.lower()

    @staticmethod
    def tokenisasi(source):
        return source.split()

    @staticmethod
    def filtering(source):
        return [word for word in source if word not in Preprocessing.stopwords]

    @staticmethod
    def stemming(source):
        # create stemmer
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

        return [stemmer.stem(word) for word in source if word and word.isascii()]

    @staticmethod
    def type(source):
        return list(OrderedDict((word, None) for word in source).keys())

    @staticmethod
    def preprocess(source):
        return Preprocessing.stemming(
            Preprocessing.filtering(
                Preprocessing.tokenisasi(
                    Preprocessing.case_folding(
                        Preprocessing.cleaning(source)
                    )
                )
            )
        )
