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
import os
import pandas
import tempfile
import urllib
import zipfile
import arff

def fetch(dataset, data_home=None, convert_strings_to_integers = True):
    if data_home is None:
        data_home = os.path.join('~', 'lale_data')
    data_home = os.path.expanduser(data_home)
    base_url = 'https://www.cs.ubc.ca/labs/beta/Projects/autoweka/datasets'
    zip_url = '{}/{}.zip'.format(base_url, dataset)
    data_dir = os.path.join(data_home, 'auto_weka', dataset)
    train_file = os.path.join(data_dir, 'train.arff')
    test_file = os.path.join(data_dir, 'test.arff')
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print('created directory {}'.format(data_dir))
        with tempfile.NamedTemporaryFile(suffix=".zip") as zip_file:
            urllib.request.urlretrieve(zip_url, zip_file.name)
            with zipfile.ZipFile(zip_file.name) as myzip:
                if not os.path.exists(train_file):
                    myzip.extract('train.arff', data_dir)
                if not os.path.exists(test_file):
                    myzip.extract('test.arff', data_dir)
    assert os.path.exists(train_file) and os.path.exists(test_file)
    def col_name(attributes, i):
        name, typ = attributes[i]
        #TODO: this currently only works for categoricals
        assert type(typ) is list #e.g., ['vhigh', 'high', 'med', 'low']
        return name
    def col_type(attributes, i):
        name, typ = attributes[i]
        #TODO: this currently only works for categoricals
        assert type(typ) is list #e.g., ['vhigh', 'high', 'med', 'low']
        return typ
    def col_list(data, i):
        return [row[i] for row in data]
    def col_list_strings_as_integers(data, i, col_type):
        from sklearn.preprocessing import LabelEncoder
        if type(col_type) is list: # categorical such as, ['vhigh', 'high', 'med', 'low']
            le = LabelEncoder()
            le.fit(col_type)
            return [le.transform([row[i]])[0] for row in data]
        else:
            return [row[i] for row in data]
    def make_X(data_dict):
        attributes, data = data_dict['attributes'], data_dict['data']
        indices = range(len(attributes) - 1)
        if convert_strings_to_integers:
            dict_of_lists = {col_name(attributes, i): col_list_strings_as_integers(data, i, col_type(attributes, i))
                            for i in indices}
        else:
            dict_of_lists = {col_name(attributes, i): col_list(data, i)
                            for i in indices}
        return pandas.DataFrame(dict_of_lists)
    def make_y(data_dict):
        attributes, data = data_dict['attributes'], data_dict['data']
        i = len(attributes) - 1
        return pandas.Series(col_list(data, i), name=col_name(attributes, i))
    with open(train_file) as f:
        train_dict = arff.load(f)
    train_X, train_y = make_X(train_dict), make_y(train_dict)
    with open(test_file) as f:
        test_dict = arff.load(f)
    test_X, test_y = make_X(test_dict), make_y(test_dict)
    return (train_X, train_y), (test_X, test_y)

def fetch_car(data_home=None, convert_strings_to_integers = True):
    return fetch('car', data_home, convert_strings_to_integers)

