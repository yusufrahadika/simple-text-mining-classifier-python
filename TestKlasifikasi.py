import datetime
import os
from Klasifikasi import Klasifikasi


class TestKlasifikasi:
    @staticmethod
    def main():
        directory_path = input("Masukkan directory_path tujuan: ")
        path_folders = os.listdir(directory_path)

        data_uji_folder_name = "Data uji"
        data_latih_folder_name = "Data latih"

        if data_latih_folder_name in path_folders and data_uji_folder_name in path_folders:
            print('start time =', datetime.datetime.now())
            klasifikasi = Klasifikasi()

            file_latih_names, file_latih_classes = TestKlasifikasi\
                .get_file_names_and_classes_from_path(directory_path + '/' + data_latih_folder_name)

            klasifikasi.train(file_latih_names, file_latih_classes)

            file_uji_names, file_uji_actual_classes = TestKlasifikasi\
                .get_file_names_and_classes_from_path(directory_path + '/' + data_uji_folder_name)

            hasil_test_classes = klasifikasi.test(file_uji_names, method_code=1)
            print(hasil_test_classes)
            akurasi = Klasifikasi.hitungAkurasi(hasil_test_classes, file_uji_actual_classes)

            print('end time =', datetime.datetime.now())
            print('akurasi =', akurasi)

        else:
            print("Path direktori tidak tepat")

    @staticmethod
    def get_file_names_and_classes_from_path(path):
        data_classes = os.listdir(path)

        file_names = []
        file_classes = []

        for data_class in data_classes:
            if not data_class.startswith('._'):
                class_path = path + '/' + data_class
                file_names_in_class = os.listdir(class_path)
                for file_name_in_class in file_names_in_class:
                    if not file_name_in_class.startswith('._'):
                        file_names.append(class_path + '/' + file_name_in_class)
                        file_classes.append(data_class)

        return file_names, file_classes


TestKlasifikasi.main()
