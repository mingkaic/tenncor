import unittest
import numpy as np

def _normalize_shape(arr):
    shape = np.trim_zeros(np.array(arr.shape) - 1) + 1
    if len(shape) > 0:
        return arr.reshape(*shape)
    return arr

class ArrTest(unittest.TestCase):
    def _array_eq(self, arr1, arr2):
        arr1 = _normalize_shape(np.array(arr1))
        arr2 = _normalize_shape(np.array(arr2))
        msg = 'diff arrays:\n{}\n{}'.format(arr1, arr2)
        self.assertTrue(np.array_equal(arr1, arr2), msg)

    def _array_close(self, arr1, arr2):
        arr1 = _normalize_shape(np.array(arr1))
        arr2 = _normalize_shape(np.array(arr2))
        msg = 'vastly diff arrays:\n{}\n{}'.format(arr1, arr2)
        self.assertTrue(np.allclose(arr1, arr2, atol=1e-05), msg)
