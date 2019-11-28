from datetime import datetime
import os
import pickle
from Klasifikasi import Klasifikasi


class TestPickledKlasifikasi:
    @staticmethod
    def main():
        model_path = input("Masukkan pickled model klasifikasi: ")

        directory_path = input("Masukkan directory path tujuan: ")
        path_folders = os.listdir(directory_path)

        data_uji_folder_name = "Data uji"

        if data_uji_folder_name in path_folders and os.path.isfile(model_path):
            print('start time =', datetime.now())
            pickle_in = open(model_path, "rb")
            klasifikasi = pickle.load(pickle_in)
            pickle_in.close()

            file_uji_names, file_uji_actual_classes = Klasifikasi \
                .get_file_names_and_classes_from_path(directory_path + '/' + data_uji_folder_name)

            hasil_test_classes = klasifikasi.test(file_uji_names)
            print(hasil_test_classes)
            akurasi = Klasifikasi.hitungAkurasi(hasil_test_classes, file_uji_actual_classes)

            print('end time =', datetime.now())
            print('akurasi =', akurasi)
        else:
            print("Path direktori tidak tepat")


TestPickledKlasifikasi.main()
