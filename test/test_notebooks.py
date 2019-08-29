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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.import subprocess
import tempfile
import unittest
import os


class TestNotebooks(unittest.TestCase):
   pass


def create_test(path):
    def exec_notebook(self):
        with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
            args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                    "--ExecutePreprocessor.timeout=1000",
                    "--output", fout.name, path]
            subprocess.check_call(args)
    return exec_notebook

for filename in os.listdir('examples'):
    if filename.lower().endswith('.ipynb'):
        test_name = 'test_notebook_{0}'.format(filename[:-len('.ipynb')])
        test_method = create_test('examples/'+filename)
        setattr(TestNotebooks, test_name, test_method)
