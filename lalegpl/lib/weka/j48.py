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
import lalegpl.lib.weka.util
from lalegpl.lib.weka.util import sklearn_input_to_weka, weka_output_to_sklearn
import lale.operators
import weka.classifiers
import pandas as pd
class J48_Impl:
    def __init__(self, **hyperparams):
        self._hyperparams = hyperparams

    def weka_options(self):
        result = []
        props = _hyperparams_schema['allOf'][0]['properties']
        for hp, value in self._hyperparams.items():
            default = props[hp]['default']
            if value != default:
                name = '-' + hp.replace('_', '-')
                result.append(name)
                if type(value) is not bool:
                    result.append(str(value))        
        return result

    def fit(self, X, y):
        options = self.weka_options()
        self._weka_model = weka.classifiers.Classifier(
            classname='weka.classifiers.trees.J48', options=options)
        if isinstance(X, pd.DataFrame):
          X = X.values
        instances, labels = sklearn_input_to_weka(X, y)
        self._weka_model.build_classifier(instances)
        self._labels = labels
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
          X = X.values
        instances, _ = sklearn_input_to_weka(
            X, labels=self._labels)
        distributions = self._weka_model.distributions_for_instances(instances)
        result = weka_output_to_sklearn(distributions)
        return result

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
          X = X.values
        instances, _ = sklearn_input_to_weka(
            X, labels=self._labels)
        distributions = self._weka_model.distributions_for_instances(instances)
        return distributions

_input_schema_fit = {
  '$schema': 'http://json-schema.org/draft-04/schema#',
  'description': 'Input data schema for training.',
  'type': 'object',
  'required': ['X', 'y'],
  'additionalProperties': False,
  'properties': {
    'X': {
      'description': 'Features; the outer array is over samples.',
      'type': 'array',
      'items': {'type': 'array', 'items': {'anyOf':[{'type': 'number'}, {'type':'string'}]}}},
    'y': {
      'description': 'Target class labels; the array is over samples.',
      'type': 'array',
      'items': {'anyOf':[{'type': 'number'}, {'type':'string'}]}}}}

_input_schema_predict = {
  '$schema': 'http://json-schema.org/draft-04/schema#',
  'description': 'Input data schema for predictions.',
  'type': 'object',
  'required': ['X'],
  'additionalProperties': False,
  'properties': {
    'X': {
      'description': 'Features; the outer array is over samples.',
      'type': 'array',
      'items': {'type': 'array', 'items': {'type': 'number'}}}}}

_output_schema = {
  '$schema': 'http://json-schema.org/draft-04/schema#',
  'description': 'Output data schema for predictions (target class labels).',
  'anyOf': [
    { 'description': 'For predict, class label.',
      'type': 'array',
      'items': { 'type': 'number'}},
    { 'description':
        'For predict_proba, for each sample, vector of probabilities.',
      'type': 'array',
      'items': { 'type': 'array', 'items': { 'type': 'number' }}}]}

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
        'U', 'O', 'C', 'M', 'R', 'N', 'B', 'S', 'L', 'A', 'J', 'Q',
        'doNotMakeSplitPointActualValue', 'output_debug_info',
        'do_not_check_capabilities', 'num_decimal_places', 'batch_size'],
      'relevantToOptimizer': [
        'U', 'C', 'M', 'R', 'N', 'B', 'S', 'L', 'A'],
      'properties': {
        'U': {
          'description': 'Use unpruned tree.',
          'type': 'boolean',
          'default': False},
        'O': {
          'description': 'Do not collapse tree.',
          'type': 'boolean',
          'default': False},
        'C': {
          'description': 'Set confidence threshold for pruning.',
          'type': 'number',
          'default': 0.25,
          'minimum': 0.01, 'exclusiveMinimum': True, #TODO: set it back to 0.
          'maximum': 1, 'exclusiveMaximum': True,
          #The code says values above 0.5 are stupid: https://svn.cms.waikato.ac.nz/svn/weka/trunk/weka/src/main/java/weka/classifiers/trees/j48/Stats.java
          'maximumForOptimizer': 0.5},
        'M': {
          'description': 'Set minimum number of instances per leaf.',
          'type': 'integer',
          'default': 2,
          'minimum': 1,
          'maximumForOptimizer': 1000},
        'R': {
          'description': 'Use reduced error pruning.',
          'type': 'boolean',
          'default': False},
        'N': {
          'description':
            'Set number of folds for reduced error pruning. '
            'One fold is used as pruning set.',
          'type': 'integer',
          'default': 3,
          'minimum': 2,
          'maximumForOptimizer': 100},
        'B': {
          'description': 'Use binary splits only.',
          'type': 'boolean',
          'default': False},
        'S': {
          'description': 'Do not perform subtree raising.',
          'type': 'boolean',
          'default': False},
        'L': {
          'description': 'Do not clean up after the tree has been built.',
          'type': 'boolean',
          'default': False},
        'A': {
          'description': 'Laplace smoothing for predicted probabilities.',
          'type': 'boolean',
          'default': False},
        'J': {
          'description':
            'Do not use MDL correction for info gain on numeric attributes.',
          'type': 'boolean',
          'default': False},
        'Q': {
          'description': 'Seed for random data shuffling.',
          'type': 'integer',
          'default': 1},
        'doNotMakeSplitPointActualValue': {
          'description': 'Do not make split point actual value.',
          'type': 'boolean',
          'default': False},
        'output_debug_info': {
          'description':
            'If set, classifier is run in debug mode and may output '
            'additional info to the console.',
          'type': 'boolean',
          'default': False},
        'do_not_check_capabilities': {
          'description':
            'If set, classifier capabilities are not checked before '
            'classifier is built (use with caution).',
          'type': 'boolean',
          'default': False},
        'num_decimal_places': {
          'description':
            'The number of decimal places for the output of numbers '
            'in the model (default 2).',
          'type': 'integer',
          'default': 2,
          'minimum': 0,
          'maximumForOptimizer': 20},
        'batch_size': {
          'description': 'The desired batch size for batch prediction.',
          'type': 'integer',
          'default': 100,
          'minimum': 1}}},
    { 'description':
        'This second sub-object lists cross-argument constraints, used '
        'to check or search conditional hyperparameters.',
      'allOf': [
        { 'description':
            "Subtree raising doesn't need to be unset for unpruned tree.",
          'anyOf': [
            { 'type': 'object',
              'properties': { 'U': {'enum': [False]}}},
            { 'type': 'object',
              'properties': {'S': {'enum': [False]}}}]},
        { 'description':
            "Unpruned tree and reduced error pruning can't be "
            "selected simultaneously.",
          'anyOf': [
            { 'type': 'object',
              'properties': { 'U': {'enum': [False]}}},
            { 'type': 'object',
              'properties': {'R': {'enum': [False]}}}]},
        { 'description':
            "Setting the confidence doesn't make sense "
            "for reduced error pruning.",
          'anyOf': [
            { 'type': 'object',
              'properties': { 'R': {'enum': [False]}}},
            { 'type': 'object',
              'properties': {'C': {'enum': [0.25]}}}]},
        { 'description':
            "Doesn't make sense to change confidence for unpruned tree.",
          'anyOf': [
            { 'type': 'object',
              'properties': { 'U': {'enum': [False]}}},
            { 'type': 'object',
              'properties': {'C': {'enum': [0.25]}}}]},
        { 'description':
            "Setting the number of folds doesn't make sense if "
            "reduced error pruning is not selected.",
          'anyOf': [
            { 'type': 'object',
              'properties': { 'R': {'enum': [True]}}},
            { 'type': 'object',
              'properties': {'N': {'enum': [3]}}}]}]}]}

_combined_schemas = {
  '$schema': 'http://json-schema.org/draft-04/schema#',
  'description': 'Combined schema for expected data and hyperparameters.',
  'documentation_url': 'http://weka.sourceforge.net/doc.dev/weka/classifiers/trees/J48.html',
  'type': 'object',
  'properties': {
    'input_fit': _input_schema_fit,
    'input_predict': _input_schema_predict,
    'output': _output_schema,
    'hyperparams': _hyperparams_schema } }

if __name__ == "__main__":
    lale.helpers.validate_is_schema(_combined_schemas)

J48 = lale.operators.make_operator(J48_Impl, _combined_schemas)
