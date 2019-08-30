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
import jsonschema
import warnings
import unittest

from lale.lib.lale import ConcatFeatures
from lale.lib.lale import NoOp
from lale.lib.sklearn import KNeighborsClassifier
from lale.lib.sklearn import LinearSVC
from lale.lib.sklearn import LogisticRegression
from lale.lib.sklearn import MinMaxScaler
from lale.lib.sklearn import Normalizer
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

from lale.search.SMAC import get_smac_space, lale_trainable_op_from_config

import numpy as np
from typing import List

def test_f_min(op, X, y, num_folds=5):
    from sklearn import datasets
    from lale.helpers import cross_val_score
    import numpy as np

    # try:
    scores = cross_val_score(op, X, y, cv = num_folds)

    return 1-np.mean(scores)  # Minimize!
    # except BaseException as e:
    #     print(e)
    #     return 

def test_iris_f_min(op, num_folds=5):
    from sklearn import datasets

    iris = datasets.load_iris()
    return test_f_min(op, iris.data, iris.target, num_folds = num_folds)

def test_iris_f_min_for_folds(num_folds=5):
    return lambda op: test_iris_f_min(op, num_folds=num_folds)
    
from lale.search.SMAC import lale_op_smac_tae

def test_iris_fmin_tae(op, num_folds=5):
    return lale_op_smac_tae(op, test_iris_f_min_for_folds(num_folds=num_folds))
        
class DontTestCar(unittest.TestCase):

    def dont_test_car_hyperopt(self):

        from lale.datasets.auto_weka import fetch_car
        from sklearn.metrics import accuracy_score, make_scorer
        from sklearn.preprocessing import LabelEncoder
        import pandas as pd
        from lale.lib.weka import J48
        from lalegpl.lib.r import ArulesCBAClassifier 
        from lale.operators import make_pipeline
        from lale.lib.lale import HyperoptClassifier
        from lale.lib.sklearn import LogisticRegression, KNeighborsClassifier

        (X_train, y_train), (X_test, y_test) = fetch_car()
        y_name = y_train.name
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        y_train = pd.Series(y_train, name=y_name)
        y_test = pd.Series(y_test, name=y_name)

        planned_pipeline = make_pipeline(ArulesCBAClassifier() | LogisticRegression() | KNeighborsClassifier())

        clf = HyperoptClassifier(model = planned_pipeline, max_evals = 1)
        best_pipeline = clf.fit(X_train, y_train)
        print(accuracy_score(y_test, best_pipeline.predict(X_test)))

    def dont_test_car_smac(self):
        import numpy as np

        from lale.datasets.auto_weka import fetch_car
        from sklearn.metrics import accuracy_score, make_scorer
        from sklearn.preprocessing import LabelEncoder
        import pandas as pd
        from lale.lib.weka import J48
        from lalegpl.lib.r import ArulesCBAClassifier 
        from lale.operators import make_pipeline
        from lale.lib.lale import HyperoptClassifier
        from lale.lib.sklearn import LogisticRegression, KNeighborsClassifier
        from smac.scenario.scenario import Scenario
        from smac.facade.smac_facade import SMAC
        from smac.configspace import ConfigurationSpace


        (X_train, y_train), (X_test, y_test) = fetch_car()
        y_name = y_train.name
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        y_train = pd.Series(y_train, name=y_name)
        y_test = pd.Series(y_test, name=y_name)

#        planned_pipeline = make_pipeline(J48() | ArulesCBAClassifier() | LogisticRegression() | KNeighborsClassifier())
        planned_pipeline = make_pipeline(ArulesCBAClassifier() | KNeighborsClassifier() | LogisticRegression())

        cs:ConfigurationSpace = get_smac_space(planned_pipeline)
        print(cs)
#        X_train = X_train[0:20]
#        y_train = y_train[0:20]
        # Scenario object
        run_count_limit = 1
        scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                            "runcount-limit": run_count_limit,  # maximum function evaluations
                            "cs": cs,               # configuration space
                            "deterministic": "true",
                            "abort_on_first_run_crash": False
                            })

        # Optimize, using a SMAC-object
        def f_min(op): 
            return test_f_min(op, X_train, y_train, num_folds=2)
        tae = lale_op_smac_tae(planned_pipeline, f_min)

        print("Optimizing! Depending on your machine, this might take a few minutes.")
        smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
                tae_runner=tae)

        incumbent = smac.optimize()
        trainable_pipeline = lale_trainable_op_from_config(planned_pipeline, incumbent)
        trained_pipeline = trainable_pipeline.fit(X_train, y_train)
        pred = trained_pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        print("Accuracy: %.2f" % (accuracy))
        inc_value = tae(incumbent)

        print("Optimized Value: %.2f" % (inc_value))
        print(f"Run count limit: {run_count_limit}")

