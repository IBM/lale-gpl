# Copyright 2021 IBM Corporation
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

import lale.operators
import numpy as np

from lale.lib.sklearn import (
    PCA,
    DecisionTreeClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    KNeighborsClassifier,
    KNeighborsRegressor,
    LinearRegression,
    LogisticRegression,
    MLPClassifier,
    LinearSVC,
    MinMaxScaler,
    Normalizer,
    Nystroem,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
)

from lale.lib.xgboost import XGBClassifier
from lale.lib.lightgbm import LGBMClassifier

from lale.lib.lale import Project, ConcatFeatures, GridSearchCV, Hyperopt, NoOp
from lale.lib.lale import OptimizeLast, Hyperopt
from lalegpl.lib.lale import NSGA2
from sklearn.metrics import get_scorer, make_scorer, matthews_corrcoef
from sklearn.model_selection import StratifiedShuffleSplit

#import logging
#logging.basicConfig(level=logging.INFO)

# Routine for computing False Positive Rate (FPR)
def compute_fpr(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = round(fp / (fp + tn), 4)
    return fpr


class TestNSGA2(unittest.TestCase):
    def setUp(self):
        import lale.datasets.openml
        import pandas as pd
        (self.X_train, self.y_train), (self.X_test, self.y_test) = lale.datasets.openml.fetch(
            'credit-g', 'classification', preprocess=True, astype='pandas')

        #print(f'train_X.shape {self.X_train.shape}')
        #print(f'test_X.shape {self.X_test.shape}')
        #print(pd.concat([self.y_train.tail(), self.X_train.tail()], axis=1))


    def test_using_individual_operator(self):

        clf_list = [DecisionTreeClassifier,
                    RandomForestClassifier,
                    ExtraTreesClassifier,
                    KNeighborsClassifier,
                    LogisticRegression,
                    GradientBoostingClassifier,
                    LGBMClassifier,
                    XGBClassifier,
                    LinearSVC,
                    MLPClassifier]

        # Now let's use NSGA2 MO optimizer to optimize the classifier
        fpr_scorer = make_scorer(compute_fpr, greater_is_better=False)

        for clf in clf_list:
            print(f"\ntest_using_individual_operator: Testing with {clf}")
            nsga2_args = {'estimator': clf, 'scoring': ['accuracy', fpr_scorer],
                          'best_score': [1, 0], 'cv': 2,
                          'max_evals': 20, 'population_size': 10}
            opt = NSGA2(**nsga2_args)
            res = opt.fit(self.X_train, self.y_train)

            df_summary = res.summary()
            print(df_summary)
            self.assertTrue(df_summary.shape[0] > 0, msg="No valid pipeline found")
            # check if summary contains valid loss values
            valid_objs = True
            for i in range(df_summary.shape[0]):
                record = df_summary.iloc[i]
                valid_objs = valid_objs and \
                             all([0 <= record['loss1'], record['loss1'] <= 1,
                              0 <= record['loss2'], record['loss2'] <= 1])
            self.assertTrue(valid_objs, msg="Invalid loss values in summary")

            acc_scorer = get_scorer('accuracy')
            _ = res.predict(self.X_test)
            pareto_pipeline = res.get_pipeline()
            print(f'test_using_individual_operator: \n'
                  'ACC, FPR scorer values on test split - %.3f %.3f' % (
                acc_scorer(pareto_pipeline, self.X_test, self.y_test),
                fpr_scorer(pareto_pipeline, self.X_test, self.y_test)))


    def test_using_pipeline(self):
        import lale.datasets.openml
        import pandas as pd
        (X_train, y_train), (X_test, y_test) = lale.datasets.openml.fetch(
            'credit-g', 'classification', preprocess=False)

        project_nums = Project(columns={'type': 'number'})
        project_cats = Project(columns={'type': 'string'})
        planned_pipeline = (
                (project_nums >> (Normalizer | NoOp) & project_cats >> OneHotEncoder)
                >> ConcatFeatures
                >> (LGBMClassifier | GradientBoostingClassifier))

        # Let's first use Hyperopt to find the best pipeline
        opt = Hyperopt(estimator=planned_pipeline, max_evals=3)
        # run optimizer
        res = opt.fit(X_train, y_train)
        best_pipeline = res.get_pipeline()

        # Now let's use NSGA2 to perform multi-objective
        # optimization on the last step (i.e., classifier)
        # in the best pipeline returned by Hyperopt
        fpr_scorer = make_scorer(compute_fpr, greater_is_better=False)
        nsga2_args = {'scoring': ['roc_auc', fpr_scorer],
                          'best_score': [1, 0], 'cv': 3,
                          'max_evals': 20, 'population_size': 10}
        opt_last = OptimizeLast(estimator=best_pipeline, last_optimizer=NSGA2,
                                         optimizer_args=nsga2_args)

        res_last = opt_last.fit(X_train, y_train)
        df_summary = res_last.summary()
        print(df_summary)
        self.assertTrue(df_summary.shape[0] > 0)

        # check if summary contains valid loss values
        valid_objs = True
        for i in range(df_summary.shape[0]):
            record = df_summary.iloc[i]
            valid_objs = valid_objs and \
                         all([0 <= record['loss1'], record['loss1'] <= 1,
                              0 <= record['loss2'], record['loss2'] <= 1])
        self.assertTrue(valid_objs, msg="Invalid loss values in summary")

        _ = res_last.predict(X_test)
        best_pipeline2 = res_last.get_pipeline()
        self.assertEqual(type(best_pipeline), type(best_pipeline2))

        auc_scorer = get_scorer('roc_auc')
        print(f'test_using_pipeline: \n'
              'AUC, FPR scorer values on test split - %.3f %.3f' % (
                  auc_scorer(best_pipeline2, X_test, y_test),
                  fpr_scorer(best_pipeline2, X_test, y_test)))

    def test_get_named_pipeline(self):
        pipeline = MinMaxScaler() >> KNeighborsClassifier()
        trained_pipeline = pipeline.fit(self.X_train, self.y_train)

        fpr_scorer = make_scorer(compute_fpr, greater_is_better=False)
        nsga2_args = {'scoring': ['accuracy', fpr_scorer],
                          'best_score': [1, 0], 'cv': 3,
                          'max_evals': 20, 'population_size': 10}
        opt_last = OptimizeLast(estimator=trained_pipeline, last_optimizer=NSGA2,
                                         optimizer_args=nsga2_args)

        res_last = opt_last.fit(self.X_train, self.y_train)

        df_summary = res_last.summary()
        pareto_pipeline = res_last.get_pipeline(pipeline_name='p0')
        self.assertEqual(type(trained_pipeline), type(pareto_pipeline))

        if (df_summary.shape[0] > 1):
            pareto_pipeline = res_last.get_pipeline(pipeline_name='p1')
            self.assertEqual(type(trained_pipeline), type(pareto_pipeline))

    def test_unspecified_arguments(self):
        fpr_scorer = make_scorer(compute_fpr, greater_is_better=False)
        nsga2_args = {'scoring': ['accuracy', fpr_scorer], 'max_evals': 10}
        opt = NSGA2(**nsga2_args)  # No arguments
        res = opt.fit(self.X_train, self.y_train)
        print(res.summary())
        _ = res.predict(self.X_test)
        best_pipeline = res.get_pipeline()

        self.assertEqual(type(best_pipeline), lale.operators.TrainedIndividualOp)


    def test_cv_object(self):
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=2, random_state=600, shuffle=True)
        print(f"Testing with Cross Validation object - {cv}")

        clf = LGBMClassifier()
        fpr_scorer = make_scorer(compute_fpr, greater_is_better=False)
        nsga2_args = {'estimator': clf,
                      'scoring': ['accuracy', fpr_scorer],
                      'best_score': [1, 0], 'cv': cv,
                      'max_evals': 20, 'population_size': 10}
        opt = NSGA2(**nsga2_args)
        res = opt.fit(self.X_train, self.y_train)
        print(res.summary())
        _ = res.predict(self.X_test)

    def test_no_crossvalidation(self):
        print(f"Testing without Cross Validation")

        clf = LGBMClassifier()
        fpr_scorer = make_scorer(compute_fpr, greater_is_better=False)
        # no CV will be performed, only single train/test split
        cv = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        nsga2_args = {'estimator': clf,
                      'scoring': ['accuracy', fpr_scorer],
                      'best_score': [1, 0], 'cv': cv,
                      'max_evals': 20, 'population_size': 10}
        opt = NSGA2(**nsga2_args)
        res = opt.fit(self.X_train, self.y_train)
        df_summary = res.summary()
        print(df_summary)
        self.assertTrue(df_summary.shape[0] > 0, msg="No valid pipeline found")

        # check if summary contains valid loss values
        valid_objs = True
        for i in range(df_summary.shape[0]):
            record = df_summary.iloc[i]
            valid_objs = valid_objs and \
                         all([0 <= record['loss1'], record['loss1'] <= 1,
                              0 <= record['loss2'], record['loss2'] <= 1])
        self.assertTrue(valid_objs, msg="Invalid loss values in summary")

        acc_scorer = get_scorer('accuracy')
        _ = res.predict(self.X_test)
        pareto_pipeline = res.get_pipeline()
        print(f'test_no_crossvalidation : Using {clf}: \n'
              'ACC, FPR scorer values on test split - %.3f %.3f' % (
                  acc_scorer(pareto_pipeline, self.X_test, self.y_test),
                  fpr_scorer(pareto_pipeline, self.X_test, self.y_test)))


    def test_opt_time_limit(self):
        import time

        clf = LGBMClassifier()
        fpr_scorer = make_scorer(compute_fpr, greater_is_better=False)
        max_opt_time = 10  # in secs
        nsga2_args = {'estimator': clf,
                      'scoring': ['accuracy', fpr_scorer],
                      'best_score': [1, 0], 'cv': 3,
                      'max_evals': 50, 'max_opt_time': max_opt_time,
                      'population_size': 10}
        opt = NSGA2(**nsga2_args)
        start = time.time()
        _ = opt.fit(self.X_train, self.y_train)
        end = time.time()
        opt_time = end - start
        rel_diff = (opt_time - max_opt_time) / max_opt_time
        print("Max time: {}, Actual time: {}, relative diff: {}".format(
            max_opt_time, opt_time, rel_diff
        ))
        assert(
                rel_diff < 0.3
        ), "Max time: {}, Actual time: {}, relative diff: {}".format(
            max_opt_time, opt_time, rel_diff
        )


    def test_invalid_args(self):
        import jsonschema

        clf = LGBMClassifier()
        nsga2_args = {'estimator': clf,
                      'cv': 3,
                      'max_evals': 50,
                      'population_size': 10}

        # No scorer specified
        with self.assertRaises(jsonschema.exceptions.ValidationError):
            _ = NSGA2(**nsga2_args)

        # Less scorers provided
        with self.assertRaises(AssertionError):
            _ = NSGA2(scoring=['accuracy'], **nsga2_args)

        # Specify LALE Pipeline as estimator. It should raise
        # AssertionError as MOO over pipelines is not supported
        pipeline = MinMaxScaler() >> KNeighborsClassifier()
        fpr_scorer = make_scorer(compute_fpr, greater_is_better=False)
        with self.assertRaises(AssertionError):
            _ = NSGA2(estimator=pipeline, scoring=['accuracy', fpr_scorer])


if __name__ == '__main__':
    unittest.main()
