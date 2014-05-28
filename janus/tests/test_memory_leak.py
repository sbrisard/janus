import gc

import pytest

import janus.greenop as greenop

from janus.discretegreenop import truncated
from janus.matprop import IsotropicLinearElasticMaterial as Material

def truncated_factory():
    return truncated(greenop.create(Material(1.0, 0.3, 2)),
                     (128, 128),
                     1.)

def num_memoryviews():
    gc.collect()
    return sum(1 if 'memoryview' in str(type(o)) else 0
               for o in gc.get_objects())

@pytest.mark.parametrize('factory, num_iters', [(truncated_factory, 10)])
def test_memory_leak(factory, num_iters):
    '''Shows that Issue #5 is solved.'''
    before = num_memoryviews()
    for i in range(num_iters):
        factory()
    assert num_memoryviews() == before
