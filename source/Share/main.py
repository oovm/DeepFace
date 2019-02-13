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


def getEvaluator():
    return 'a evaluator'


def getModel():
    return 'a model'


def Saver(data, path):
    return 'a model'


class DeepFace:
    __loaded = False
    __generator = False
    __modifier = False

    __Model = None
    __Evaluator = None

    __recorder = ()

    def __init__(self, model, dataset, ctx=mx.cpu()):
        (self.__Model, self.dataset) = load_check(model, dataset)
        self.ctx = ctx

    def load(self, dir=default_dir()):
        # do loading

        self.__Evaluator = getEvaluator(self.__Model)
        self.__Model = getModel(self.__Model, self.dataset, dir=dir)

        self.__loaded = True

    def new(self, num=1):
        if not self.__loaded:
            raise ValueError("The model haven't loaded.")
        if not self.__generator:
            raise ValueError("The model is not a Generator!")
        self.__recorder = self.__Evaluator(self.__Model, num)
        return self.__recorder

    def infer(self, *args):
        if not self.__loaded:
            raise ValueError("The model haven't loaded.")
        if not self.__modifier:
            raise ValueError("The model is not a Modifier!")
        self.__recorder = self.__Evaluator(self.__Model, args)
        return self.__recorder

    def save(self, path):
        Saver(self.__recorder, path)


