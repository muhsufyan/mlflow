from package.feature.data_processing import get_feature_dataframe

from package.feature.split_data import get_train_test_score_set
from package.ml_training.train import train_model
from package.feature.preprocessing_pipeline import get_pipeline

from package.utils.utils import set_or_create_experiment
from package.utils.utils import get_performance_plots
from package.utils.utils import get_classification_metrics
from package.utils.utils import register_model_with_client
import mlflow

if __name__ == "__main__":
    # setup untuk mlflow experiment
    experiment_name = "house_pricing_classifier_2"
    run_name = "training_classifier_2"
    model_name = "registered_model_2"
    artifact_path = "model_2"
    
    # get dataset, proses EDA di skip karena mlflow disini hanya fokus dibagian modeling
    # untuk EDA lakukan dg dvc
    df = get_feature_dataframe()

    # split dataset
    x_train, x_test, x_score, y_train, y_test, y_score = get_train_test_score_set(df)

    # feature id, target dan MedHouseVal dari dataset tdk diambil
    features = [f for f in x_train.columns if f not in ["id", "target", "MedHouseVal"]]

    # buat pipeline dataset (data preprocessing yg kita buat)
    pipeline = get_pipeline(numerical_features=features, categorical_features=[])

    # buat mlflow experiment
    experiment_id = set_or_create_experiment(experiment_name=experiment_name)

    # lakukan training model
    run_id, model = train_model(pipeline=pipeline, run_name=run_name, model_name=model_name, artifact_path=artifact_path, x=x_train[features], y=y_train)

    # lakukan inference/prediksi 
    y_pred = model.predict(x_test)

    # matriks evaluasi
    classification_metrics = get_classification_metrics(
        y_true=y_test, y_pred=y_pred, prefix="test"
    )

    # simpan dalam gambar hsl matriks evaluasinya
    performance_plots = get_performance_plots(
        y_true=y_test, y_pred=y_pred, prefix="test"
    )
    
    # model registry
    # mlflow.register_model(model_uri=f"runs:/{run_id}/{artifact_path}", name=model_name) # jika kode ini di run lagi (misal 2 kali dijlnkan)
    # maka akan dibuatkan register model yg baru tp versionnya berbeda, misal 2 kali dijlnkan maka akan ada version 1 dan version 2
    # kita buat registry model secara manual menggunakan registered model client dg kode dibawah ini (kode pembuatan registry model diatas hrs di non aktifkan dulu)
    # intinya hrs dipilih salah satu kode untuk registry model
    # register_model_with_client(model_name=model_name, run_id=run_id, artifact_path=artifact_path)
    # dg menjlnkan kode registry model scra manual ini, ketika dijnkan 2 kali maka akan muncul pesan error
    # hal ini berbeda dg mlflow.register_model sblmnya 
    # kedua kode untuk registry model tdk perlu diaktifkan karena di bagian train.py sdh dimasukkan

    # jlnkan mlflow run
    # log performance metrics
    with mlflow.start_run(run_id=run_id):
        # log metrics
        mlflow.log_metrics(classification_metrics)

        # log params
        mlflow.log_params(model[-1].get_params())

        # log tags
        mlflow.set_tags({"type": "classifier"})

        # log description
        mlflow.set_tag(
            "mlflow.note.content", "This is a classifier for the house pricing dataset"
        )

        # log plots (gambar matriks evaluasi)
        for plot_name, fig in performance_plots.items():
            mlflow.log_figure(fig, plot_name + ".png")