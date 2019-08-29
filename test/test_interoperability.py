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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.import unittest
import warnings
import random
import jsonschema
import sys
import lale.operators as Ops
from lale.lib.lale import ConcatFeatures
from lale.lib.lale import NoOp
from lale.lib.sklearn import KNeighborsClassifier
from lale.lib.sklearn import LinearSVC
from lale.lib.sklearn import LogisticRegression
from lale.lib.sklearn import MinMaxScaler
from lale.lib.sklearn import MLPClassifier
from lale.lib.sklearn import Nystroem
from lale.lib.sklearn import OneHotEncoder
from lale.lib.sklearn import PCA
from lale.lib.sklearn import TfidfVectorizer
from lale.lib.sklearn import MultinomialNB
from lale.lib.sklearn import SimpleImputer
from lale.lib.sklearn import SVC
from lale.lib.xgboost import XGBClassifier
from lale.lib.sklearn import PassiveAggressiveClassifier
from lale.lib.sklearn import StandardScaler
from lale.lib.sklearn import FeatureAgglomeration
from lalegpl.lib.weka import J48
from typing import List

import sklearn.datasets

from lale.sklearn_compat import make_sklearn_compat
from lale.search.GridSearchCV import LaleGridSearchCV, get_grid_search_parameter_grids
from lale.search.SMAC import get_smac_space, lale_trainable_op_from_config
from lale.search.op2hp import hyperopt_search_space


def tearDownModule():
    print('running test.test.tearDownModule()')
    import weka.core.jvm
    if weka.core.jvm.started:
        print('test.test.tearDownModule() is stopping the JVM')
        weka.core.jvm.stop()

def test_in_subprocess(cls):
    import sys
    cls_name = '{}.{}'.format(cls.__module__, cls.__name__)
    need_subprocess = (cls_name not in sys.argv)
    if need_subprocess:
        import subprocess
        class Wrapper(unittest.TestCase):
            def test_wrapper(self):
                argv = ['python', '-m', 'unittest', '-v', cls_name]
                print('subprocess({}) starting'.format(argv))
                subprocess.check_call(argv)
                print('subprocess({}) completed'.format(argv))
        return Wrapper
    else:
        return cls

class TestArulesCBAClassifier(unittest.TestCase):
    def test_hyperparam_defaults(self):
        from lalegpl.lib.r import ArulesCBAClassifier as planned
        trainable = planned()
    def test_init_fit_predict(self):
        from lalegpl.lib.r import ArulesCBAClassifier as planned
        trainable = planned(support=0.05, confidence=0.9)
        from lalegpl.datasets.auto_weka import fetch_car
        (X_train, y_train), (X_test, y_test) = fetch_car(convert_strings_to_integers = True)

        y_name = y_train.name
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        import pandas as pd
        import numpy as np
        y_train = pd.Series(y_train, name=y_name, dtype=np.int)
        y_test = pd.Series(y_test, name=y_name, dtype=np.int)

        trained = trainable.fit(X_train, y_train)
        predictions = trained.predict(X_test)
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, predictions)
        print(f'accuracy {accuracy:.1%}')


@test_in_subprocess
class TestJ48test(unittest.TestCase):
    def test_hyperparam_defaults(self):
        trainable = J48()
        hyperopt_search_space(trainable)

@test_in_subprocess
class TestJ48(unittest.TestCase):
    def test_hyperparam_defaults(self):
        trainable = J48()
    def test_init_fit_predict(self):
        trainable = J48(U=True)
        iris = sklearn.datasets.load_iris()
        trained = trainable.fit(iris.data, iris.target)
        predicted = trained.predict(iris.data)
    def test_pipeline_iris_pca_j48(self):
        import sklearn.datasets
        import sklearn.utils
        iris = sklearn.datasets.load_iris()
        X_all, y_all = sklearn.utils.shuffle(
            iris.data, iris.target, random_state=42)
        holdout_size = 20
        X_train, y_train = X_all[holdout_size:], y_all[holdout_size:]
        X_test, y_test = X_all[:holdout_size], y_all[:holdout_size]
        import lale.helpers
        pca = PCA(n_components=3)
        j48 = J48()
        trainable_pipe = pca >> j48
        print('before calling fit on pipeline')
        trained_pipe = trainable_pipe.fit(X_train, y_train)
        print('after calling fit on pipeline')
        lale.helpers.to_graphviz(trained_pipe)
        predicted = trained_pipe.predict(X_test)
    def test_pipeline_digits_scaler_j48(self):
        import sklearn.datasets
        import sklearn.utils
        digits = sklearn.datasets.load_digits()
        X_all, y_all = sklearn.utils.shuffle(
            digits.data, digits.target, random_state=42)
        holdout_size = 200
        X_train, y_train = X_all[holdout_size:], y_all[holdout_size:]
        X_test, y_test = X_all[:holdout_size], y_all[:holdout_size]
        from lale.lib.sklearn import MinMaxScaler
        import lale.helpers
        scaler = MinMaxScaler()
        j48 = J48()
        trainable_pipe = scaler >> j48
        print('before calling fit on pipeline')
        trained_pipe = trainable_pipe.fit(X_train, y_train)
        print('after calling fit on pipeline')
        lale.helpers.to_graphviz(trained_pipe)
        predicted = trained_pipe.predict(X_test)
    def test_J48_for_car_dataset(self):
        from lalegpl.datasets.auto_weka import fetch_car
        (X_train, y_train), (X_test, y_test) = fetch_car()
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        clf = J48()
        from sklearn.metrics import accuracy_score
        from lale.lib.lale import NoOp, HyperoptClassifier
        from lale.operators import make_pipeline
        clf = HyperoptClassifier(make_pipeline(J48()), max_evals = 1)
        trained_clf = clf.fit(X_train, y_train)
        print(accuracy_score(y_test, trained_clf.predict(X_test)))

