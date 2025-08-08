import anomaly_new as an
import predict as pre

file_path = input("Dosyanın konumunu giriniz: ").strip('"').strip("'")
an.system(file_path)

while True:
 choice = int(input("Model eğitildi. Yeniden kullanmak için 1 tuşuna, programı sonlandırmak için 0 tuşuna basınız."))
 if choice == 1 :
  file_path = input("Dosyanın konumunu giriniz: ").strip('"').strip("'")
  pre.system(file_path)
 else:
  print("Programdan çıkılıyor.")
  break