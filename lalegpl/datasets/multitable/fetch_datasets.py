# Copyright 2019 IBM Corporation
#
# Licensed under the GNU General Public License 3.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.gnu.org/licenses/gpl-3.0.txt
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import csv
import logging
import os
import urllib.request

import mysql.connector
import pandas as pd

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from pyspark.sql import SparkSession

    spark_installed = True
except ImportError:
    spark_installed = False

imdb_config = {
    "user": "guest",
    "password": "relational",
    "host": "relational.fit.cvut.cz",
    "database": "imdb_ijs",
    "port": 3306,
    "raise_on_warnings": True,
}


def get_data_from_csv(datatype, data_file_name):
    if datatype.casefold() == "pandas":
        return pd.read_csv(data_file_name)
    elif datatype.casefold() == "spark":
        if spark_installed:
            spark = SparkSession.builder.appName("GoSales Dataset").getOrCreate()
            return spark.read.csv(data_file_name, header=True)
        else:
            raise ValueError("Spark is not installed on this machine.")
    else:
        raise ValueError(
            "Can fetch the go_sales data in pandas or spark dataframes only. Pass either 'pandas' or 'spark' in datatype parameter."
        )


def fetch_imdb_dataset(datatype="pandas"):

    """
    Fetches the IMDB movie dataset from Relational Dataset Repo.
    It contains information about directors, actors, roles
    and genres of multiple movies in form of 7 CSV files.
    This method downloads and stores these 7 CSV files under the
    'lale/lale/datasets/multitable/imdb_data' directory. It creates
    this directory by itself if it does not exists.

    Dataset URL: https://relational.fit.cvut.cz/dataset/IMDb

    Parameters
    ----------
    datatype : string, optional, default 'pandas'

      If 'pandas',
      Returns a list of singleton dictionaries (each element of the list is one
      table from the dataset) after reading the downloaded / existing CSV files.
      The key of each dictionary is the name of the table and the value contains
      a pandas dataframe consisting of the data.

      If 'spark',
      Returns a list of singleton dictionaries (each element of the list is one
      table from the dataset) after reading the downloaded / existing CSV files.
      The key of each dictionary is the name of the table and the value contains
      a spark dataframe consisting of the data.

      Else,
      Throws an error as it does not support any other return type.

    Returns
    -------
    imdb_list : list of singleton dictionary of pandas / spark dataframes
    """

    try:
        cnx = mysql.connector.connect(**imdb_config)
        cursor = cnx.cursor()
        imdb_table_list = []
        download_data_dir = os.path.join(os.path.dirname(__file__), "imdb_data")
        imdb_list = []
        cursor.execute("show tables")
        for table in cursor:
            imdb_table_list.append(table[0])
        for table in imdb_table_list:
            header_list = []
            cursor.execute("desc {}".format(table))
            for column in cursor:
                header_list.append(column[0])
            csv_name = "{}.csv".format(table)
            data_file_name = os.path.join(download_data_dir, csv_name)
            if not os.path.exists(data_file_name):
                if not os.path.exists(download_data_dir):
                    os.makedirs(download_data_dir)
                cursor.execute("select * from {}".format(table))
                result = cursor.fetchall()
                file = open(data_file_name, "w", encoding="utf-8")
                c = csv.writer(file)
                c.writerow(header_list)
                for row in result:
                    c.writerow(row)
                file.close()
                logger.info(" Created:{}".format(data_file_name))
            imdb_list.append(
                {csv_name.split(".")[0]: get_data_from_csv(datatype, data_file_name)}
            )
        logger.info(" Fetched the IMDB dataset. Process completed.")
        return imdb_list
    except mysql.connector.Error as err:
        raise ValueError(err)
    else:
        cnx.close()
