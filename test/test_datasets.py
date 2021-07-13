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

import unittest

import jsonschema
import pandas as pd

try:
    from pyspark import SparkConf, SparkContext
    from pyspark.sql import Row, SparkSession, SQLContext

    spark_installed = True
except ImportError:
    spark_installed = False

from lale.lib.lale import Join
from lale.expressions import it
from lalegpl.datasets.multitable.fetch_datasets import fetch_imdb_dataset

# Testing join operator for pandas dataframes
class TestJoin(unittest.TestCase):
    def test_init(self):
        _ = Join(pred=[it.main.train_id == it.info.TrainId], join_type="inner")

    # TestCase 1: IMDB dataset
    def test_join_pandas_imdb(self):
        imdb = fetch_imdb_dataset()
        trainable = Join(
            pred=[
                it.movies_directors.movie_id == it.movies_genres.movie_id,
                it.movies_genres.movie_id == it.movies.id,
                it.movies_directors.movie_id == it.roles.movie_id,
            ],
            join_type="left",
        )
        transformed_df = trainable.transform(imdb)
        self.assertEqual(transformed_df.shape, (6062848, 9))
        self.assertEqual(transformed_df["movie_id"][1], 281325)


# Testing join operator for spark dataframes
class TestJoinSpark(unittest.TestCase):
    # TestCase 1: IMDB dataset
    def test_join_spark_imdb(self):
        if spark_installed:
            imdb = fetch_imdb_dataset("spark")
            trainable = Join(
                pred=[
                    it.movies_directors.movie_id == it.movies_genres.movie_id,
                    it.movies_genres.movie_id == it.movies.id,
                    it.movies_directors.movie_id == it.roles.movie_id,
                ],
                join_type="left",
            )
            transformed_df = trainable.transform(imdb)
            self.assertEqual(transformed_df.count(), 6062848)
            self.assertEqual(len(transformed_df.columns), 9)
