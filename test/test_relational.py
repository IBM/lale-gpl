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

import logging
import time
import unittest

import jsonschema
import pandas as pd

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from pyspark import SparkConf, SparkContext
    from pyspark.sql import Row, SparkSession, SQLContext

    spark_installed = True
except ImportError:
    spark_installed = False

from lale.lib.lale import Join, Filter
from lale.expressions import it
from lale.datasets.data_schemas import get_table_name
from lalegpl.datasets.multitable.fetch_datasets import fetch_imdb_dataset


# Testing join operator for pandas dataframes
class TestBenchmarkJoinAndFilterPandas(unittest.TestCase):
    def test_benchmark_pandas(self):
        imdb = fetch_imdb_dataset()

        start_time = time.time()
        trainable = Join(pred=[it.movies_directors.movie_id == it.movies.id],join_type="inner")
        transformed_df = trainable.transform(imdb)
        trainable = Filter(pred=[it["director_id"] == 8])
        filtered_df = trainable.transform(transformed_df)
        self.assertEqual(filtered_df.shape, (35, 6))
        join_first = time.time() - start_time
        logger.info(" Pandas: Join Before Filter --- {} seconds".format(join_first))

        movies_directors = imdb[4]
        self.assertEqual(get_table_name(movies_directors), "movies_directors")
        start_time = time.time()
        trainable = Filter(pred=[it["director_id"] == 8])
        filtered_df = trainable.transform(movies_directors)
        self.assertEqual(get_table_name(filtered_df), "movies_directors")
        imdb.pop(4)
        imdb.append(filtered_df)
        trainable = Join(pred=[it.movies_directors.movie_id == it.movies.id],join_type="inner")
        transformed_df = trainable.transform(imdb)
        self.assertEqual(transformed_df.shape, (35, 6))
        filter_first = time.time() - start_time
        logger.info(" Pandas: Join After Filter --- {} seconds".format(filter_first))


class TestBenchmarkJoinAndFilterSpark(unittest.TestCase):
    def test_benchmark_join_before_filter_spark(self):
        if spark_installed:
            imdb = fetch_imdb_dataset("spark")
            start_time = time.time()
            trainable = Join(pred=[it.movies_directors.movie_id == it.movies.id],join_type="inner")
            transformed_df = trainable.transform(imdb)
            trainable = Filter(pred=[it["director_id"] == 8])
            filtered_df = trainable.transform(transformed_df)
            self.assertEqual(filtered_df.count(), 35)
            self.assertEqual(len(filtered_df.columns), 6)
            return time.time() - start_time

    def test_benchmark_join_after_filter_spark(self):
        if spark_installed:
            imdb = fetch_imdb_dataset("spark")
            movies_directors = imdb[4]
            self.assertEqual(get_table_name(movies_directors), "movies_directors")
            start_time = time.time()
            trainable = Filter(pred=[it["director_id"] == 8])
            filtered_df = trainable.transform(movies_directors)
            self.assertEqual(get_table_name(filtered_df), "movies_directors")
            imdb.pop(4)
            imdb.append(filtered_df)
            trainable = Join(pred=[it.movies_directors.movie_id == it.movies.id],join_type="inner")
            transformed_df = trainable.transform(imdb)
            self.assertEqual(transformed_df.count(), 35)
            self.assertEqual(len(transformed_df.columns), 6)
            return time.time() - start_time

    def test_benchmark_spark(self):
        if spark_installed:
            filter_first = self.test_benchmark_join_after_filter_spark()
            logger.info(" Spark: Join After Filter --- {} seconds".format(filter_first))
            join_first = self.test_benchmark_join_before_filter_spark()
            logger.info(" Spark: Join Before Filter --- {} seconds".format(join_first))
            self.assertLessEqual(join_first - join_first / 5, filter_first)
            self.assertLessEqual(filter_first, join_first + join_first / 5)
