from Weighting import Weighting

import os
from Preprocessing import Preprocessing


class TestWeighting:
    @staticmethod
    def main():
        directory = input('Masukkan directory folder txt: ')

        weighting = Weighting()
        weighting.setText([
            Preprocessing.stemming(
                Preprocessing.filtering(
                    Preprocessing.tokenisasi(
                        Preprocessing.case_folding(
                            Preprocessing.cleaning(open(directory + '/' + filename, 'r', encoding="ISO-8859-1").read())
                        )
                    )
                )
            )
            for filename in os.listdir(directory)
            if filename.lower().endswith('.txt') and not filename.startswith('._')
        ])

        print(weighting.getFeatures())
        print(weighting.getTf())
        print(weighting.getTIdf())


TestWeighting.main()
