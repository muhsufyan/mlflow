from sklearn.datasets import fetch_california_housing
import pandas as pd
import os 

def load_data() -> pd.DataFrame:
    # work path
    cur_directory_path = os.path.abspath(os.path.dirname(__file__))
    # read dataset. kasus ini dataset adalah california housing
    data = fetch_california_housing(
        data_home=f"{cur_directory_path}/data/", as_frame=True, download_if_missing=True
    )
    return data.frame


def get_feature_dataframe() -> pd.DataFrame:
    df = load_data()
    df["id"] = df.index
    # feature output/targetnya. MedHouseVal : median of the house value for each district. 
    # dg demikian kasus sbnrnya adlh regresi
    df["target"] = df["MedHouseVal"] >= df["MedHouseVal"].median() # filter berdsrkan MedHouseVal
    df["target"] = df["target"].astype(int)
 
    return df