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
import rpy2.robjects
import rpy2.robjects.packages
import lale.helpers

def install_r_package(pkg_name):
    if 'R_LIBS_USER' in os.environ:
        lib_dir = os.environ['R_LIBS_USER']
    else:
        lib_dir = os.path.expanduser(os.path.join('~', 'R', 'lib'))
    if not os.path.exists(lib_dir):
        os.makedirs(lib_dir)
    libPaths_fun = rpy2.robjects.r['.libPaths']
    libPaths_fun(lib_dir)
    rutils = rpy2.robjects.packages.importr('utils')
    #rutils.chooseCRANmirror(ind=1)
    rutils.chooseCRANmirror()
    if not rpy2.robjects.packages.isinstalled(pkg_name):
        lale.helpers.println_pos(f'installing R package {pkg_name} to libPaths {libPaths_fun()}')
        rutils.install_packages(pkg_name)
    assert rpy2.robjects.packages.isinstalled(pkg_name), f'failed to install {pkg_name}'
    pkg = rpy2.robjects.packages.importr(pkg_name)
    return pkg

def create_r_dataframe(X, y=None):
    if not isinstance(X, pandas.DataFrame):
        X = pandas.DataFrame(X, columns=['f'+str(i) for i in range(X.shape[1])], index=[i for i in range(X.shape[0])])
    if y is not None and not isinstance(y, pandas.Series):
        y = pandas.Series(y, name="target")

    def create_r_vec(pd_df, col_name):
        col = pd_df[col_name]
        #TODO: make this code work for other types besides categorical strings
        str_vec = rpy2.robjects.IntVector(col) #This will work for car dataset for now.
        return str_vec
    col_rvecs_X = {name: create_r_vec(X, name) for name in X.columns}
    if y is None:
        col_rvecs_y = {}
    else:
        col_rvecs_y = {y.name: rpy2.robjects.FactorVector(rpy2.robjects.IntVector(y))}
    col_rvecs_all = {**col_rvecs_X, **col_rvecs_y}
    result = rpy2.robjects.DataFrame(col_rvecs_all)
    return result
