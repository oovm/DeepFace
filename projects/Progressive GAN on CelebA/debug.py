import pickle

import PIL.Image
import numpy as np
import tensorflow as tf
import wolframclient.serializers as wxf

name = 'karras2018iclr-celebahq-1024x1024'
file = open(name + '.pkl', 'rb')
sess = tf.InteractiveSession()
G, D, Gs = pickle.load(file)

# Generate latent vectors.
latents = np.random.RandomState(1000).randn(1000, *Gs.input_shapes[0][1:])  # 1000 random latents
latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]]  # hand-picked top-10
wxf.export(latents, 'latents.mx', target_format='wxf')

# Generate dummy labels (not used by the official networks).
labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

# Run the generator to produce a set of images.
images = Gs.run(latents, labels)

# Convert images to PIL-compatible format.
images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
wxf.export(latents, 'debug.mx', target_format='wxf')

# Save images as PNG.
images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC
for idx in range(images.shape[0]):
    PIL.Image.fromarray(images[idx], 'RGB').save('img%d.png' % idx)
