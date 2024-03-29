from datetime import datetime
import os
from Klasifikasi import Klasifikasi


class TestKlasifikasi:
    @staticmethod
    def main():
        directory_path = input("Masukkan directory path tujuan: ")
        path_folders = os.listdir(directory_path)

        data_uji_folder_name = "Data uji"
        data_latih_folder_name = "Data latih"

        if data_latih_folder_name in path_folders and data_uji_folder_name in path_folders:
            print('start time =', datetime.now())
            klasifikasi = Klasifikasi()

            file_latih_names, file_latih_classes = Klasifikasi\
                .get_file_names_and_classes_from_path(directory_path + '/' + data_latih_folder_name)

            klasifikasi.train(file_latih_names, file_latih_classes)

            file_uji_names, file_uji_actual_classes = Klasifikasi\
                .get_file_names_and_classes_from_path(directory_path + '/' + data_uji_folder_name)

            hasil_test_classes = klasifikasi.test(file_uji_names)
            print(hasil_test_classes)
            akurasi = Klasifikasi.hitungAkurasi(hasil_test_classes, file_uji_actual_classes)

            print('end time =', datetime.now())
            print('akurasi =', akurasi)

        else:
            print("Path direktori tidak tepat")


TestKlasifikasi.main()
