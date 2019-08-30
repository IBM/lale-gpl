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
import rpy2.robjects
import rpy2.robjects.packages
import numpy as np

class ArulesCBAClassifier_Impl:
    def __init__(self, **hyperparams):
        self._hyperparams = hyperparams

    def fit(self, X, y):
#        assert type(X) is pandas.DataFrame and type(y) is pandas.Series
        arules_pkg = install_r_package('arulesCBA')
        formula = rpy2.robjects.Formula('{} ~ .'.format(y.name))
        r_train = create_r_dataframe(X, y)
        hps = {k: v for k, v in self._hyperparams.items() if v is not None}
        self._r_model = arules_pkg.CBA(formula=formula, data=r_train, **hps)
        return self

    def predict(self, X):
        assert type(X) is pandas.DataFrame
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
      'items': {'type': 'array', 'items': {'type': 'integer'}}},
    'y': {
      'description': 'Target class labels; the array is over samples.',
      'type': 'array',
      'items': {'type': 'integer'}}}}

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
      'items': {'type': 'array', 'items': {'type': 'integer'}}}}}

_output_schema = {
  '$schema': 'http://json-schema.org/draft-04/schema#',
  'description':
    'Output data schema for predictions (target class labels). '
    'So far only works for strings.',
  'type': 'array',
  'items': {'type': 'integer'}}

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
        'support', 'confidence', 'verbose', 'parameter', 'control',
        'sort_parameter', 'lhs_support', 'disc_method'],
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
        'verbose': {
          'description':
            'Optional logical flag to allow verbose execution, where '
            'additional intermediary execution information is printed '
            'at runtime.',
          'type': 'boolean',
          'default': False},
        'parameter': {
          'description': 'Optional parameter list for apriori.',
          'anyOf': [
            { 'type': 'array'},
            { 'enum': [None]}],
          'default': None},
        'control': {
          'description': 'Optional control list for apriori.',
          'anyOf': [
            { 'type': 'array'},
            { 'enum': [None]}],
          'default': None},
        'sort_parameter': {
          'description':
            'Ordered vector of arules interest measures (as characters) '
            'which are used to sort rules in preprocessing.',
          'anyOf': [
            { 'type': 'array'},
            { 'enum': [None]}],
          'default': None},
        'lhs_support': {
          'description':
            'Logical variable, which, when set to True, indicates that '
            'LHS support should be used for rule mining.  LHS support '
            'rule mining is considerably slower than normal mining.',
          'type': 'boolean',
          'default': False},
        'disc_method': {
          'description':
            'Discretization method for factorizing numeric input.',
          'default': 'mdlp',
          'enum':
            ['mdlp', 'caim', 'cacc', 'ameva', 'chi2', 'chimerge',
             'extendedchi2', 'modchi2']}}}]}

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
    'output': _output_schema,
    'hyperparams': _hyperparams_schema } }

if __name__ == "__main__":
    lale.helpers.validate_is_schema(_combined_schemas)

from lale.operators import make_operator
ArulesCBAClassifier = make_operator(ArulesCBAClassifier_Impl, _combined_schemas)
