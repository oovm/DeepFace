# DeepFace.load(model=None,dataset=None)

import mxnet as mx

from .download import default_dir


# model = 'PGGAN'

# dataset = 'CelebA'


def load_check(model, dataset):
    if model is None:
        return 'PGGAN', 'CelebA'

    if model is 'PGGAN':
        if dataset is None:
            return 'PGGAN', 'CelebA'
        else:
            return model, dataset

    if model is 'SBGAN':
        if dataset is None:
            return 'SBGAN', 'CelebA'
        else:
            return model, dataset


print(load_check(None, None))
print(load_check('SBGAN', None))

generators = ['PGGAN', 'SBGAN']


def get_evaluator():
    return 'a evaluator'


def get_model():
    return 'a model'


def Saver(data, path):
    return 'a model'


class DeepFace:
    __loaded = False
    __Evaluator = None
    recorder = []

    def __init__(self, model=None, dataset=None, ctx=mx.cpu()):
        (self.model, self.dataset) = load_check(model, dataset)
        self.ctx = ctx

    def load(self, dir=default_dir()):
        # do loading

        if self.model in generators:
            return DeepFaceGenerator(
                get_model(self.model, self.ctx),
                get_evaluator(self.model)
            )
        else:
            return DeepFaceModifier(
                get_model(self.model, self.ctx),
                get_evaluator(self.model)
            )

        self.__loaded = True

    def save(self, path):
        Saver(self.recorder, path)


class DeepFaceGenerator(DeepFace):
    __loaded = False
    module = None
    __Evaluator = None

    recorder = ()

    def __init__(self, m, e):
        self.module = m
        self.__Evaluator = e

    def new(self, num=1):
        if not self.__loaded:
            raise ValueError("The model haven't loaded.")
        if not self.__generator:
            raise ValueError("The model is not a Generator!")
        self.__recorder = self.__Evaluator(self.module, num)
        return self.__recorder


class DeepFaceModifier(DeepFace):
    __loaded = False
    module = None
    __Evaluator = None
    recorder = ()

    def __init__(self, m, e):
        self.module = m
        self.__Evaluator = e

    def infer(self, *args):
        if not self.__loaded:
            raise ValueError("The model haven't loaded.")
        if not self.__modifier:
            raise ValueError("The model is not a Modifier!")
        self.__recorder = self.__Evaluator(self.module, args)
        return self.__recorder
