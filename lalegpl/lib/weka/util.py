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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.# See also https://github.com/fracpete/python-weka-wrapper3
import javabridge
import lale.helpers
import numpy as np
import re
import weka.core.classes
import weka.core.jvm
import pandas as pd

if not weka.core.jvm.started:
    import logging
    weka.core.jvm.logger.setLevel(logging.WARNING)
    weka.core.jvm.start()

def sklearn_input_to_weka(X, y=None, labels=None):
    from weka.core.dataset import Attribute, Instances, Instance
    attribs = []
    for i in range(len(X[0])):
        attribs.append(Attribute.create_numeric('x_{}'.format(i)))
    if labels is None and y is not None:
        labels = [str(label) for label in np.unique(y)]
    attribs.append(Attribute.create_nominal('y', labels))
    n_rows = len(X)
    instances = Instances.create_instances('data', attribs, n_rows)
    for i in range(n_rows):
        if y is None:
            row = [*X[i], '0']
        elif isinstance(y, pd.Series):
            row = [*X[i], y.iloc[i]]
        else:
            row = [*X[i], y[i]]
        instances.add_instance(Instance.create_instance(row))
    instances.class_is_last()
    return instances, labels

def weka_output_to_sklearn(dists):
    return np.argmax(dists, axis=1)

def options(weka_model):
    jobj = weka_model.jobject
    opts = javabridge.call(jobj, 'listOptions', '()Ljava/util/Enumeration;')
    enum = javabridge.get_enumeration_wrapper(opts)
    return enum

def hyperparam_ranges(weka_model):
    opts = options(weka_model)
    ranges = {}
    while opts.hasMoreElements():
        opt = weka.core.classes.Option(opts.nextElement())
        name = opt.name.strip('-\t\n ').replace('-', '_')
        assert name not in ranges
        description = opt.description.strip()
        if opt.num_arguments == 0:
            ranges[name] = {
                'description': description,
                'type': 'boolean',
                'default': False}
        else:
            match = re.search(r'\(default ([^)]+)\)', description)
            try:
                default = int(match[1])
                typ = 'integer'
            except ValueError:
                default = float(match[1])
                typ = 'number'
            ranges[name] = {
                'description': description,
                'type': typ,
                'default': default,
                'minimum': 0,
                'maximum': 1000}
    return ranges
