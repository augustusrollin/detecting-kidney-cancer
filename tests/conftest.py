import pytest


@pytest.fixture()
def data_folder():
    folder = "kidney_ct_data"
    folder = "sample_kidney_ct_data"
    return folder

@pytest.fixture()
def csv_path():
    csv = "kidneyData.csv"
    csv = "sample_kidneyData.csv"
    return csv

