import os
import pickle

import tensorflow as tf
import wolframclient.serializers as wxf

name = 'karras2018iclr-celebahq-1024x1024'
file = open(name + '.pkl', 'rb')
sess = tf.InteractiveSession()
G, D, Gs = pickle.load(file)
saver = tf.train.Saver()
save_path = "./tmp/" + name + "/"
model_name = 'model'
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_path_full = os.path.join(save_path, model_name)
saver.save(sess, save_path_full)

ckpt = tf.train.get_checkpoint_state(save_path)
reader = tf.train.NewCheckpointReader(ckpt.model_checkpoint_path)
all_variables = list(reader.get_variable_to_shape_map().keys())
npy = dict(zip(all_variables, map(reader.get_tensor, all_variables)))
# remove `float32` because it had not be supported
npy.pop('D_paper/lod')
npy.pop('G_paper/lod')
npy.pop('G_paper_1/lod')

wxf.export(npy, name + '.mx', target_format='wxf')
