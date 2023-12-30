# buat environment 
python -m venv env

# masuk ke environment
# jika di windows
.\env\Scripts\activate
# jika di linux/mac
source .\env\bin\activate

# latihan ini akan dibungkus dlm 1 package. cara buat package nya
mkdir package\package
cd package
touch setup.py
mkdir package
# direktori untuk feature dan training model
mkdir feature ml_training

# buat file yg handle preprocessing
cd feature
touch data_preprocessing.py

cd ..

# buat file yg handle training
cd ml_training
touch train.py

# tes
cd ../../..
touch run.py

# buat package
cd package
pip install build
python -m build 
# maka akan mengenerate direktori dist (distribution) yg mrpkn package buatan kita sendiri
# untuk menggunakan package nya jlnkan perintah brkt (tp sblmnya kita lihat perbedaan dg pip list)
cd dist 
# install package yg kita buat. caranya dg menginstall file berekstensi .whl
pip install package-1.0.0-py3-none-any.whl --force-reinstall
# pip list dan lihat bagian package, inilah package yg telah kita buat sendiri

# tes memanggil package yg tlh kita buat
cd ../..
python .\run.py

# penambahan file di feature sehingga file di direktori feature ada data_preprocessing.py, preprocessing.py, dan split_data.py
# kode untuk di file train.py pd direktori ml_training
# pembuatan direktori baru yaitu utils dg file utils.py
# dibagian setup.py bagian install_requires dinon aktifkan

# lalu jlnkan lagi perintah
cd package
python -m build 
cd dist
pip install package-1.0.0-py3-none-any.whl --force-reinstall
cd ../..
# diterminal yg lainnya jlnkan mlflow webserver
mlflow ui
# jlnkan proses preprocessing, training, evaluasi model serta mlflow tracking
python .\run.py

# di root work directory, buat file yg akan memprediksi/inference dalam batches
touch batch_inference.py
python .\batch_inference.py

# di root work dir, buat file yg akan memprediksi/inference scra online/real time/langsung
touch online_inference.py

# untuk deploy model menggunakan mlflow maka gunakan perintah
mlflow models serve --model-uri models:/registered_model/latest --no-conda 
# maka model akan dideploy, kasus ini local
# jlnkan testing untuk real time/online inference
python .\online_inference.py