import numpy
import theano
from nose.plugins.skip import SkipTest
from theano.tests.unittest_tools import verify_grad

try:
    from python.response_norm import (
        CrossMapNorm,
        CrossMapNormUndo
    )
    from theano.sandbox.cuda import CudaNdarrayType, CudaNdarray
    from theano.sandbox.cuda import gpu_from_host
except ImportError:
    raise SkipTest('cuda not available')


def test_cross_map_norm_simple():
    op = CrossMapNorm(16, 15. / 16., 1., True)
    x = CudaNdarray(numpy.ones((10, 32, 5, 5), dtype='float32'))
    x_ = theano.tensor.TensorVariable(CudaNdarrayType([False] * 4))
    f = theano.function([x_], op(x_)[0])
    numpy.testing.assert_allclose(f(x), 0.0625)
    print x.shape
    print (numpy.asarray(f(x))).shape

def test_cross_map_norm_grad_simple():
    rng = numpy.random.RandomState([2013, 02, 10])
    op = CrossMapNorm(16, 15/16., 1, True)
    make_graph = lambda inp: op(gpu_from_host(inp))[0]
    verify = lambda array: verify_grad(make_graph, [array])
    inputs = [numpy.ones((5, 32, 4, 4), dtype='float32'),
              rng.normal(size=(10, 32, 5, 5)).astype('float32')]
    for arr in inputs:
        yield verify, arr

def test_optimization():
    op = CrossMapNorm(16, 15./16., 1, True)
    x_ = theano.tensor.TensorVariable(CudaNdarrayType([False] * 4))
    f = theano.function([x_], theano.grad(op(x_)[0].sum(), x_))
    nodes = [x for x in f.maker.fgraph.apply_nodes
             if type(x.op) == CrossMapNormUndo]
    assert len(nodes) == 1
    assert nodes[0].op.inplace

if __name__ == '__main__':
    test_cross_map_norm_simple()
    test_cross_map_norm_grad_simple() # cannot work unless test_cross_map_norm_grad_simple().next()
    test_optimization()
    print "Successful !"
