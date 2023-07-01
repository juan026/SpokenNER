# File: data.py, Project: SpokenNer
# Created by Moncef Benaicha
# Contact: support@moncefbenaicha.me

import json
import os


class Json:
    @staticmethod
    def load(data_path: str) -> dict:
        """
            The method load takes a json path and load after verification
            the method will return the value of key 'data', that is expected to contain {train:[], valid:[], test:[]},
            if it exists otherwise the whole loaded json will be returned as dict
        :param data_path: Json file path
        :return: Dictionary of json file
        """

        if os.path.exists(data_path) and data_path.endswith(".json"):
            with open(data_path, "r") as data_source:
                dataset = json.load(data_source)
                return dataset.get("data") or dataset
        else:
            raise ValueError(
                f"Provided data path: {data_path}, is invalid. Please make sure the path exists and it's a json file"
            )
