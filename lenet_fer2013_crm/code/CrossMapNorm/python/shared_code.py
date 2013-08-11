"""
Convenience methods for accessing C code shared by the code
generators in different parts of this module.
"""

__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

import os

this_dir =  os.path.dirname(os.path.abspath(__file__)) + "/../src/"

def load_code(local_path):
    path = this_dir + local_path
    f = open(path)
    return f.read()

def get_NVMatrix_code():
    header = '#include "nvmatrix.cuh"'
    source1 = load_code("nvmatrix.cu")
    source2 = load_code("nvmatrix_kernels.cu")

    source = source1 + source2

    rval = header + source

    return rval

