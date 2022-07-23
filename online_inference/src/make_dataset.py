# -*- coding: utf-8 -*-
from typing import Tuple, NoReturn

import pandas as pd
from boto3 import client


def download_data_from_s3(
    s3_bucket: str, s3_path: str, output: str, endpoint_url="https://ib.bizmrg.com"
) -> NoReturn:
    s3 = client("s3", endpoint_url=endpoint_url)
    s3.download_file(s3_bucket, s3_path, output)


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data
