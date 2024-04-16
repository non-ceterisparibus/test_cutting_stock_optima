import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_equal
from numpy.testing import assert_almost_equal
import pytest


@pytest.fixture
def setup_stock_finish_1():
    out = {}
    out['stock'] = {
    "S1": {"width": 1219, "weight": 4395 },
    "S2": {"width": 1219, "weight": 9260},
    "S3": {"width": 1219, "weight": 3475},
    "S4": {"width": 1219, "weight": 8535},
    # "S5": {"width": 236, "weight": 1571},
    }

    out['finish'] = {
    "F1": {"width": 235, "need_cut": 11524 , "fc1":6522},
    "F2": {"width": 147, "need_cut": 1308, "fc1": 10417},
    "F3": {"width": 136, "need_cut": 1290, "fc1": 22574},
    "F4": {"width": 68, "need_cut": 309, "fc1": 4619},
    "F5": {"width": 60, "need_cut": 159, "fc1": 0},
    "F6": {"width": 85, "need_cut": 132, "fc1": 4560},
    "F7": {"width": 57, "need_cut": 100, "fc1": 104},
    "F8": {"width": 92, "need_cut": 100, "fc1": 669}, 
    # "F9": {"width": 57, "need_cut": 735, "fc1": 669}, 
    }

    return out

def setup_stock_finish_2():
    out = {}
    out['stock'] = {
    "S1": {"width": 1219, "weight": 4395 },
    "S2": {"width": 1219, "weight": 9260},
    "S3": {"width": 1219, "weight": 3475},
    "S4": {"width": 1219, "weight": 8535},
    "S5": {"width": 236, "weight": 1571},
    }

    out['finish'] = {
    "F1": {"width": 235, "need_cut": 11524 , "fc1":6522},
    "F2": {"width": 147, "need_cut": 1308, "fc1": 10417},
    "F3": {"width": 136, "need_cut": 1290, "fc1": 22574},
    "F4": {"width": 130, "need_cut": 562, "fc1": 4619},
    "F5": {"width": 123, "need_cut": 704, "fc1": 0},
    "F6": {"width": 100, "need_cut": 307, "fc1": 4560},
    "F7": {"width": 92, "need_cut": 100, "fc1": 104},
    "F8": {"width": 92, "need_cut": 318, "fc1": 669},
    "F9": {"width": 85, "need_cut": 132, "fc1": 912},
    }

    return out