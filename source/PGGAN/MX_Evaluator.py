from collections import namedtuple

import mxnet as mx

from .main import ProgressiveGan
from ..Share import Recorder

Batch = namedtuple('Batch', ['data'])

data = mx.random.normal(shape=(4, 512))


class EvaluatorMX(ProgressiveGan):
    net = None

    def download(self, *args):
        pass

    def load(self, symbol, params):
        sym = mx.symbol.load(symbol)
        mod = mx.module.Module(
            symbol=sym,
            data_names=['Input'],
            label_names=None
        )
        mod.bind(data_shapes=[('Input', (1, 512))])
        mod.load_params(params)
        self.net = mod

    def forward(self, data):
        self.net.forward(Batch([data]), is_train=False)
        out = self.net.get_outputs()[0].asnumpy()
        return Recorder(
            'Generator',
            data=out,  # FIXME:format data
            show=out
        )

    def phrase(self, input=None, dir='./', *args):
        if input is None:
            return self.phrase((1, 512))
        if isinstance(input, int):
            self.phrase((input, 512))
