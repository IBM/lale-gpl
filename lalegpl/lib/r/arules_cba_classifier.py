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
import lale
import lale.helpers
from lalegpl.lib.r.util import install_r_package, create_r_dataframe
import pandas
try:
  import rpy2.robjects
  import rpy2.robjects.packages
except ImportError:
  raise ImportError("""ArulesCBAClassifier needs a Python package called `rpy2`. 
  You can install it using `pip install rpy2` or install lalegpl[full] which will install it for you.""")

import numpy as np

class ArulesCBAClassifier_Impl:
    def __init__(self, **hyperparams):
        self._hyperparams = hyperparams

    def fit(self, X, y):
        arules_pkg = install_r_package('arulesCBA')
        if not isinstance(y, pandas.Series):
          y_name = "target"
        else:
          y_name = y.name
        formula = rpy2.robjects.Formula(f'{y_name} ~ .')
        r_train = create_r_dataframe(X, y)
        hps = {k: v for k, v in self._hyperparams.items() if v is not None}
        if False:
            lale.helpers.println_pos('arules_pkg.CBA(formula="{}", data=[\n{}], {})'.format(formula, r_train, ', '.join([f'{k}={v}' for k, v in hps.items()])))
        self._r_model = arules_pkg.CBA(formula=formula, data=r_train, **hps)
        return self

    def predict(self, X):
        stats_pkg = rpy2.robjects.packages.importr('stats')
        predict_fun = stats_pkg.predict
        r_test = create_r_dataframe(X)
        r_predictions = predict_fun(self._r_model, r_test)
        levels = r_predictions.levels
        predictions = [levels[yi - 1] for yi in r_predictions]
        return np.array(predictions, dtype=np.int)

_input_schema_fit = {
  '$schema': 'http://json-schema.org/draft-04/schema#',
  'description':
    'Input data schema for training. So far only works for strings.',
  'type': 'object',
  'required': ['X', 'y'],
  'additionalProperties': False,
  'properties': {
    'X': {
      'description': 'Features; the outer array is over samples.',
      'type': 'array',
      'items': {'type': 'array', 'items': {'type': 'number'}}},
    'y': {
      'description': 'Target class labels; the array is over samples.',
      'type': 'array',
      'items': {'type': 'number'}}}}

_input_schema_predict = {
  '$schema': 'http://json-schema.org/draft-04/schema#',
  'description':
    'Input data schema for predictions. So far only works for strings',
  'type': 'object',
  'required': ['X'],
  'additionalProperties': False,
  'properties': {
    'X': {
      'description': 'Features; the outer array is over samples.',
      'type': 'array',
      'items': {'type': 'array', 'items': {'type': 'number'}}}}}

_output_predict_schema = {
  '$schema': 'http://json-schema.org/draft-04/schema#',
  'description':
    'Output data schema for predictions (target class labels). '
    'So far only works for strings.',
  'type': 'array',
  'items': {'type': 'number'}}

_hyperparams_schema = {
  '$schema': 'http://json-schema.org/draft-04/schema#',
  'description': 'Hyperparameter schema.',
  'allOf': [
    { 'description':
        'This first sub-object lists all constructor arguments with their '
        'types, one at a time, omitting cross-argument constraints.',
      'type': 'object',
      'additionalProperties': False,
      'required': [
        'support', 'confidence', 'disc_method', 'balanceSupport', 'pruning'],
      'relevantToOptimizer': ['confidence', 'disc_method'],
      'properties': {
        'support': {
          'description': 'Minimum support for creating association rules.',
          'type': 'number',
          'default': 0.2,
          'minimum': 0,
          'maximumForOptimizer': 1.0},
        'confidence': {
          'description': 'Minimum confidence for creating association rules.',
          'type': 'number',
          'default': 0.8,
          'minimum': 0,
          'maximumForOptimizer': 1.0},
        'disc_method': {
          'description':
            'Discretization method for factorizing numeric input.',
          'default': 'mdlp',
          'enum':
            ['mdlp', 'caim', 'cacc', 'ameva', 'chi2', 'chimerge',
             'extendedchi2', 'modchi2']},
        'balanceSupport': {
          'description': 'If true, class imbalance is counteracted by using the minimum support only for the majority class.',
          'type': 'boolean',
          'default': False},
        'pruning': {
          'enum': ['M1', 'M2'],
          'default': 'M1'},
        # 'parameter': {
        #   'description': 'Optional parameter list for apriori.',
        #   'anyOf': [
        #     { 'type': 'array'},
        #     { 'enum': [None]}],
        #   'default': None},
        # 'control': {
        #   'description': 'Optional control list for apriori.',
        #   'anyOf': [
        #     { 'type': 'array'},
        #     { 'enum': [None]}],
        #   'default': None},
        # 'verbose': {
        #   'description':
        #     'Optional logical flag to allow verbose execution, where '
        #     'additional intermediary execution information is printed '
        #     'at runtime.',
        #   'type': 'boolean',
        #   'default': False},
}}]}

_combined_schemas = {
  '$schema': 'http://json-schema.org/draft-04/schema#',
  'description': 'Combined schema for expected data and hyperparameters.',
  'documentation_url': 'https://cran.r-project.org/web/packages/arulesCBA/',
  'type': 'object',
  'tags': {
    'pre': ['categoricals'],
    'op': ['estimator', 'classifier', 'interpretable'],
    'post': []},
  'properties': {
    'input_fit': _input_schema_fit,
    'input_predict': _input_schema_predict,
    'output_predict': _output_predict_schema,
    'hyperparams': _hyperparams_schema } }

if __name__ == "__main__":
    lale.helpers.validate_is_schema(_combined_schemas)

from lale.operators import make_operator
ArulesCBAClassifier = make_operator(ArulesCBAClassifier_Impl, _combined_schemas)
