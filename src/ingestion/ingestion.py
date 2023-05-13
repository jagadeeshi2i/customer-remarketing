import os
import sys
import argparse

from pathlib import Path

from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split

# from src.components.transformation import DataTransformation
# from src.components.transformation import DataTransformationConfig


# from src.components.trainer import ModelTrainerConfig
# from src.components.trainer import ModelTrainer

class DataIngestion:
    def __init__(self,input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        Path(str(self.output_folder)).mkdir(parents=True, exist_ok=True)
        self.train_data_path =  os.path.join(self.output_folder, "train.csv")
        self.test_data_path =  os.path.join(self.output_folder, "test.csv")
        self.raw_data_path =  os.path.join(self.output_folder, "data.csv")

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            data = os.path.join(self.input_folder, "data.csv")
            df = pd.read_csv(data, sep=";")
            logging.info("Read the dataset as dataframe")


            df.to_csv(self.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42
            )

            train_set.to_csv(
                self.train_data_path, index=False, header=True
            )

            test_set.to_csv(
                self.test_data_path, index=False, header=True
            )

            logging.info("Ingestion of the data is completed")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str)
    parser.add_argument("--output_folder", type=str)

    args = parser.parse_args()
    obj = DataIngestion(args.input_folder, args.output_folder)

    obj.initiate_data_ingestion()
