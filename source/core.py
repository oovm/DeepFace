from .PGGAN import mxPGGAN, ptPGGAN
from .Share import Saver


class Generator:
    framework = None
    device = None

    __binded = False
    __loaded = False
    __Evaluator = None
    __Recorder = None

    def __init__(self, m, e):
        self.module = m
        self.__Evaluator = e

    def bind(self, framework, device):

        # if have install
        self.framework = framework

        # if have device
        self.device = device

    def load(self, model, dataset):
        if not self.__binded:
            raise ValueError("Haven't Bind!")

        if self.framework is 'mxnet':
            if model is 'PGGAN':
                self.__Evaluator = mxPGGAN(

                )

        elif self.framework is 'pytorch':
            if model is 'PGGAN':
                self.__Evaluator = ptPGGAN(

                )

        self.__loaded = True

    def new(self, *args):
        if not self.__binded:
            raise ValueError("Bind First!")
        if not self.__loaded:
            raise ValueError("The model haven't loaded.")

        self.__Recorder = self.__Evaluator.eval(args)
        return self.__Recorder.show

    def save(self, *args):
        Saver(self.__Recorder, args)


class Modifier:
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
