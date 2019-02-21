import hashlib
import os
from collections import namedtuple

import mxnet as mx

os.chdir("../../projects/Progressive GAN on CelebA")

print("Workspace: " + os.getcwd())


class Config:
    symbol = os.path.join(os.getcwd(), 'PGGAN_CelebA-symbol.json')
    params = os.path.join(os.getcwd(), 'PGGAN_CelebA-0000.params')
    ctx = mx.gpu(0)


def md5(filename, block_size=65536):
    hash = hashlib.md5()
    with open(filename, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            hash.update(block)
    return hash.hexdigest()


print("Symbol Check: " + md5(Config.symbol))
print("Params Check: " + md5(Config.params))

# Check infer
Batch = namedtuple('Batch', ['data'])
input = mx.random.normal(shape=(4, 512))
print("\n", "Input Shape:", input.shape)

sym = mx.symbol.load(Config.symbol)

mod = mx.module.Module(
    symbol=sym,
    data_names=['Input'],
    label_names=None
)
mod.bind(data_shapes=[('Input', (1, 512))])
mod.load_params(Config.params)

mod.forward(Batch([input]), is_train=False)
out = mod.get_outputs()[0]

print("Output Shape: ", out.shape)
