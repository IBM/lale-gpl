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

import logging
import time

import jsonschema
import numpy as np
import pandas as pd
try:
    from platypus import (
        HUX,
        NSGAII,
        PM,
        SBX,
        Binary,
        BitFlip,
        CompoundOperator,
        Integer,
        Problem,
        Real,
        nondominated,
    )
except ImportError:
  raise ImportError("""NSGA2 needs a Python package called `platypus`. 
  You can install it using `pip install platypus-opt>=1.0.4` or install lalegpl[full] which will install it for you.""")

from sklearn.metrics import get_scorer
from sklearn.model_selection import check_cv, train_test_split

import lale.docstrings
import lale.operators
from lale.lib.lale._common_schemas import (
    check_scoring_best_score_constraint,
    schema_best_score,
    schema_cv,
    schema_estimator,
    schema_max_opt_time,
    schema_scoring_list,
)
from lale.lib.sklearn import LogisticRegression

logger = logging.getLogger(__name__)


# Exception for handling max optimization time
class MaxBudgetExceededException(Exception):
    pass


class _ModelHelper:
    def __init__(self, model):
        (
            self.param_choices,
            self.param_type,
            self.param_categories,
        ) = self.__get_param_choices_types(model)
        type_map = {
            "number": Real,
            "integer": Integer,
            "boolean": Binary,
            "enum": Integer,
        }

        logger.info(self.param_choices)
        logger.info(self.param_type)
        logger.info(self.param_categories)

        types = []
        for key in self.param_choices:
            if self.param_type[key] == "boolean":
                types.append(type_map[self.param_type[key]](1))
            else:
                types.append(
                    type_map[self.param_type[key]](
                        self.param_choices[key][0], self.param_choices[key][1]
                    )
                )

        self.types = types
        logger.info(self.types)

        self.model = model

    def __get_param_choices_types(self, model):
        range_dict, cat_idx = model.get_param_ranges()

        param_choices = {}
        param_type = {}
        param_categories = {}
        for key in range_dict:
            if key not in cat_idx.keys():
                minval, maxval, defval = range_dict[key]
                if minval == maxval:
                    continue

                hp_schema = model.hyperparam_schema(key)
                if "type" in hp_schema.keys():
                    ptype = hp_schema["type"]
                else:
                    # ptype = hp_schema['anyOf'][0]['type']
                    if isinstance(defval, int):
                        ptype = "integer"
                    elif isinstance(defval, float):
                        ptype = "number"
                    else:
                        ptype = hp_schema["anyOf"][0]["type"]

                # if ptype == 'boolean':
                #    continue

                param_choices[key] = [minval, maxval]
                param_type[key] = ptype
            else:
                minval, maxval, defval = cat_idx[key]
                if minval == maxval:
                    continue
                param_choices[key] = [minval, maxval]
                param_type[key] = "enum"  # for categorical inputs
                param_categories[key] = range_dict[key]

        return param_choices, param_type, param_categories

    def create_instance(self, parameter):
        logger.debug("Creating model instance with params: \n" f"{parameter}")

        clf = self.model.with_params(**parameter)
        return clf


class _NSGA2Impl:
    def __init__(
        self,
        estimator=None,
        scoring=None,
        best_score=0.0,
        cv=5,
        max_evals=50,
        max_opt_time=None,
        population_size=10,
        random_seed=42,
    ):
        if estimator is None:
            self.model = LogisticRegression()
        else:
            self.model = estimator

        assert isinstance(self.model, lale.operators.IndividualOp), (
            "Multi-objective optimization is supported for only "
            "Individual Operators currently and not supported over Pipelines."
        )
        logger.info(f"Optimizing model {self.model} with type {type(self.model)}")
        logger.info("Lale param ranges - \n" f"{self.model.get_param_ranges()}")
        self.model_helper = _ModelHelper(self.model)
        self.moo_solutions = []

        self.scoring = scoring
        assert self.scoring is not None, "scoring parameter not specified."
        assert len(self.scoring) >= 2, "Less than two scorers specified in scoring"

        if isinstance(best_score, list):
            if len(best_score) < len(scoring):
                best_score.extend([0.0] * (len(scoring) - len(best_score)))
            self.best_score = best_score
        else:
            self.best_score = [best_score] * len(scoring)

        self.cv = cv
        self.max_evals = max_evals
        self.max_opt_time = max_opt_time
        self.population_size = population_size
        self.random_seed = random_seed

    @classmethod
    def validate_hyperparams(cls, scoring=None, best_score=0, **hyperparams):
        check_scoring_best_score_constraint(scoring, best_score)

    # Internal class
    class Soln(object):
        def __init__(self, variables, objectives):
            self.variables = variables
            self.objectives = objectives

    # convert parameter list to dictionary
    def param_to_dict(self, parameter, param_choices, param_categories, param_type):
        temp = {}
        i = 0
        for key in param_choices:
            if key not in param_categories.keys():  # if non-categorical parameter
                if param_type[key] == "boolean":
                    temp[key] = parameter[i][0]
                else:
                    temp[key] = parameter[i]
            else:
                temp[key] = param_categories[key][parameter[i]]

            i += 1

        return temp

    def fit(self, X, y):

        opt_start_time = time.time()
        kfold = None
        if isinstance(self.cv, int) and self.cv == 1:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=self.random_seed, stratify=y
            )
            logger.info(
                "Not using Cross-Validation. " "Performing single train/test split"
            )
        else:
            is_clf = self.model.is_classifier()
            kfold = check_cv(self.cv, y=y, classifier=is_clf)
            # kfold = StratifiedKFold(
            #    n_splits=self.cv, random_state=self.random_seed, shuffle=True
            # )
            logger.info(f"Using Cross-Validation - {kfold}")

        self.ind = 0

        def train_test_model(parameter):
            # First check if we exceeded allocated time budget
            current_time = time.time()
            elapsed_time = current_time - opt_start_time
            if (self.max_opt_time is not None) and (elapsed_time > self.max_opt_time):
                msg = (
                    f"Max optimization time exceeded. "
                    f"Max Opt time = {self.max_opt_time}, Elapsed Time = {elapsed_time}, "
                    f"NFE Completed - {self.ind}"
                )
                raise MaxBudgetExceededException(msg)

            self.ind = self.ind + 1
            logger.info(f"Training population {self.ind}")

            parameter = self.param_to_dict(
                parameter,
                self.model_helper.param_choices,
                self.model_helper.param_categories,
                self.model_helper.param_type,
            )

            scorers = [get_scorer(scorer) for scorer in self.scoring]
            nscorers = len(scorers)

            try:
                if kfold is None:
                    clf = self.model_helper.create_instance(parameter)
                    clf_trained = clf.fit(X_train, y_train)

                    obj_val = [scorer(clf_trained, X_val, y_val) for scorer in scorers]

                else:

                    obj_scores = [[] for _ in range(nscorers)]

                    # Perform k-fold cross-validation
                    for train_index, test_index in kfold.split(X, y):
                        if isinstance(X, pd.DataFrame):
                            X_train_split, X_val_split = (
                                X.iloc[train_index],
                                X.iloc[test_index],
                            )
                            y_train_split, y_val_split = (
                                y.iloc[train_index],
                                y.iloc[test_index],
                            )
                        else:
                            X_train_split, X_val_split = X[train_index], X[test_index]
                            y_train_split, y_val_split = y[train_index], y[test_index]

                        clf = self.model_helper.create_instance(parameter)
                        clf_trained = clf.fit(X_train_split, y_train_split)

                        obj_score = [
                            scorer(clf_trained, X_val_split, y_val_split)
                            for scorer in scorers
                        ]
                        for i in range(nscorers):
                            obj_scores[i].append(obj_score[i])

                    # Aggregate CV score
                    obj_val = [np.mean(obj_scores[i]) for i in range(nscorers)]
                    logger.debug(f"Obj k-fold scores - {obj_scores}")

                # By default we are solving a minimization MOO problem
                fitnessValue = [
                    self.best_score[i] - obj_val[i] for i in range(nscorers)
                ]
                logger.info(f"Train fitnessValue - {fitnessValue}")

            except jsonschema.ValidationError as e:
                logger.error(f"Caught JSON schema validation error.\n{e}")
                logger.error("Setting fitness (loss) values to infinity")
                fitnessValue = [np.inf for i in range(nscorers)]
                logger.info(f"Train fitnessValue - {fitnessValue}")

            return fitnessValue

        def time_check_callback(alg):
            current_time = time.time()
            elapsed_time = current_time - opt_start_time
            logger.info(f"NFE Complete - {alg.nfe}, Elapsed Time - {elapsed_time}")

        parameter_num = len(self.model_helper.param_choices)
        target_num = len(self.scoring)
        # Adjust max_evals if not a multiple of population size. This is
        # required as Platypus performs evaluations in multiples of
        # population_size.
        adjusted_max_evals = (
            self.max_evals // self.population_size
        ) * self.population_size
        if adjusted_max_evals != self.max_evals:
            logger.info(
                f"Adjusting max_evals to {adjusted_max_evals} from specified {self.max_evals}"
            )

        problem = Problem(parameter_num, target_num)
        problem.types[:] = self.model_helper.types
        problem.function = train_test_model

        # Set the variator based on types of decision variables
        varg = {}
        first_type = problem.types[0].__class__
        all_type_same = all([isinstance(t, first_type) for t in problem.types])
        # use compound operator for mixed types
        if not all_type_same:
            varg["variator"] = CompoundOperator(SBX(), HUX(), PM(), BitFlip())

        algorithm = NSGAII(
            problem,
            population_size=self.population_size,
            **varg,
        )

        try:
            algorithm.run(adjusted_max_evals, callback=time_check_callback)
        except MaxBudgetExceededException as e:
            logger.warning(
                f"Max optimization time budget exceeded. Optimization exited prematurely.\n{e}"
            )

        solutions = nondominated(algorithm.result)
        # solutions = [s for s in algorithm.result if s.feasible]`
        # solutions = algorithm.result

        moo_solutions = []
        for solution in solutions:
            vars = []
            for pnum in range(parameter_num):
                vars.append(problem.types[pnum].decode(solution.variables[pnum]))

            vars_dict = self.param_to_dict(
                vars,
                self.model_helper.param_choices,
                self.model_helper.param_categories,
                self.model_helper.param_type,
            )
            moo_solutions.append(self.Soln(vars_dict, solution.objectives))
            logger.info(f"{vars}, {solution.objectives}")

        self.moo_solutions = moo_solutions

        pareto_models = []
        for solution in self.moo_solutions:
            est = self.model_helper.create_instance(solution.variables)
            est_trained = est.fit(X, y)
            pareto_models.append((solution.variables, est_trained))

        self.pareto_models = pareto_models
        return self

    def get_pareto_solutions(self):
        return self.moo_solutions

    def get_pareto_models(self):
        return self.pareto_models

    # Predict using first pareto-optimal estimator
    def predict(self, X, **kwargs):
        if "pipeline_name" in kwargs:
            pname = kwargs["pipeline_name"]
            pipeline = self.get_pipeline(pipeline_name=pname)
            del kwargs["pipeline_name"]
        else:
            pipeline = self.get_pipeline()

        return pipeline.predict(X, **kwargs)

    # Return pareto-optimal estimator
    def get_pipeline(self, pipeline_name=None, astype="lale"):
        """Retrieve one of the pareto-optimal pipelines.

        Parameters
        ----------
        pipeline_name : union type, default None

            - string
                Key (name) from the table returned by summary(), return a trained pipeline.

            - None
                When not specified, return the first (trained) pipeline in the table
                returned by summary()

        astype : 'lale' or 'sklearn', default 'lale'
            Type of resulting pipeline.

        Returns
        -------
        result : Trained operator."""

        id = 0
        if pipeline_name is not None:
            id = int(pipeline_name[1:])

        assert 0 < len(self.pareto_models), "No pipelines found"
        assert id < len(self.pareto_models), "Invalid pipeline name"
        vars, pareto_model = self.pareto_models[id]
        result = pareto_model

        if astype == "lale":
            return result

        assert astype == "sklearn", "Invalid astype " + astype
        if hasattr(result, "export_to_sklearn_pipeline"):
            result = result.export_to_sklearn_pipeline()
        else:
            logger.warning("Cannot return sklearn pipeline.")

        return result

    def summary(self):
        """Table displaying the pareto-optimal solutions (pipelines)
           obtained after multi-objective optimization
           (name, ID, loss for each specified scorer).

        Returns
        -------
        result : DataFrame"""

        nsolutions = len(self.moo_solutions)
        nscoring = len(self.scoring)

        records = []

        for isol in range(nsolutions):
            record_dict = {}
            record_dict["name"] = f"p{isol}"
            record_dict["id"] = isol
            for iobj in range(nscoring):
                solution = self.moo_solutions[isol]
                record_dict[f"loss{iobj+1}"] = solution.objectives[iobj]

            records.append(record_dict)

        result = pd.DataFrame.from_records(records, index="name")
        return result


_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Multi Objective Optimizer based on NSGA-II algorithm.

    Example
    --------
    >>> import lale.datasets.openml
    >>> (X_train, y_train), (X_test, y_test) =
    ...     lale.datasets.openml.fetch('credit-g', 'classification', preprocess=True, astype='pandas')
    >>>
    >>> # Create sklearn scorer for computing FPR
    >>> def compute_fpr(y_true, y_pred):
    ...     from sklearn.metrics import confusion_matrix
    ...     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    ...     fpr = round(fp / (fp + tn), 4)
    ...     return fpr
    >>>
    >>> from sklearn.metrics import make_scorer
    >>> fpr_scorer = make_scorer(compute_fpr, greater_is_better=False)
    >>>
    >>> from lale.lib.xgboost import XGBClassifier
    >>> clf = XGBClassifier()
    >>> nsga2_args = {'estimator': clf, 'scoring': ['accuracy', fpr_scorer],
    ...               'best_score': [1, 0], 'cv': 3,
    ...               'max_evals': 20, 'population_size': 10}
    >>> opt = NSGA2(**nsga2_args)
    >>> trained = opt.fit(X_train, y_train)
    >>> # Predict using first pareto-optimal solution (pipeline)
    >>> predictions = trained.predict(X_test, pipeline_name='p0')
    >>> from sklearn.metrics import accuracy_score
    >>> acc = accuracy_score(y_test, predictions)
    >>> fpr = compute_fpr(y_test, predictions)
    >>> print('Accuracy, FPR - %.3f, %.3f' % (acc, fpr))

    """,
    "documentation_url": "https://lale-gpl.readthedocs.io/en/latest/modules/lalegpl.lib.lale.nsga2.html",
    "import_from": "lalegpl.lib.lale",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator"], "post": []},
    "properties": {
        "hyperparams": {
            "allOf": [
                {
                    "type": "object",
                    "properties": {
                        "estimator": schema_estimator,
                        "scoring": schema_scoring_list,
                        "best_score": schema_best_score,
                        "cv": schema_cv,
                        "max_evals": {
                            "description": "Number of trials of Hyperopt search.",
                            "type": "integer",
                            "minimum": 1,
                            "default": 50,
                        },
                        "max_opt_time": schema_max_opt_time,
                        "population_size": {"default": 10},
                        "random_seed": {"default": 42},
                    },
                    "additionalProperties": False,
                    "required": ["estimator", "scoring"],
                    "relevantToOptimizer": [],
                }
            ]
        },
        "input_fit": {
            "type": "object",
            "properties": {
                "X": {"laleType": "Any"},
                "y": {"laleType": "Any"},
            },
            "additionalProperties": False,
            "required": ["X", "y"],
        },
        "input_predict": {
            "type": "object",
            "properties": {
                "X": {"laleType": "Any"},
                "pipeline_name": {
                    "description": "Name of the pipeline to use for prediction",
                    "anyOf": [
                        {
                            "type": "string",
                            "description": "Which pipeline to pick.  Must be in the list returned by summary.",
                        },
                        {
                            "enum": [None],
                            "description": "Run predict on the first pipeline.",
                        },
                    ],
                },
            },
            "additionalProperties": True,
            "required": ["X"],
        },
        "output_predict": {"laleType": "Any"},
    },
}

NSGA2 = lale.operators.make_operator(_NSGA2Impl, _combined_schemas, name="NSGA2")
lale.docstrings.set_docstrings(NSGA2)