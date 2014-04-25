import janus.greenop as greenop

from janus.discretegreenop import truncated
from janus.matprop import IsotropicLinearElasticMaterial as Material

import gc

def truncated_factory():
    return truncated(greenop.create(Material(1.0, 0.3, 2)),
                     (128, 128),
                     1.)

def num_memoryviews():
    gc.collect()
    return sum(1 if 'memoryview' in str(type(o)) else 0
               for o in gc.get_objects())

def test(factory, num_iters):
    for i in range(num_iters):
        factory()
        print('Iteration {0},  {1} memoryviews.'.format(i + 1,
                                                        num_memoryviews()))

if __name__ == '__main__':
    test(truncated_factory, 10)
