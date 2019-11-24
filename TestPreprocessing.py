from Preprocessing import Preprocessing


class TestPreprocessing:
    @staticmethod
    def main():
        path = input('Masukkan path txt yang akan dibaca: ')
        if path.lower().endswith('.txt'):
            print('File: ' + path)
            file = open(path, 'r', encoding="ISO-8859-1")
            file_content = file.read()
            file.close()
            print('Hasil Cleaning:')
            result = Preprocessing.cleaning(file_content)
            print(result)
            print('Hasil Case Folding:')
            result = Preprocessing.case_folding(result)
            print(result)
            print('Hasil Tokenisasi:')
            result = Preprocessing.tokenisasi(result)
            print(result)
            print('Hasil Filtering:')
            result = Preprocessing.filtering(result)
            print(result)
            print('Hasil Stemming:')
            result = Preprocessing.stemming(result)
            print(result)
        else:
            print('File tidak valid')


TestPreprocessing().main()
