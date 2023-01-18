from Vectorizer import *
from Model import *
from datasets import load_dataset

class Classification():

    def __init__(self,vectorizer: Vectorizer, model: Model) -> None:
        self._vectorizer = vectorizer
        self._model = model

    @property
    def model(self) -> Model:
        return self._model

    @model.setter
    def model(self, model: Model) -> None:
        self._model = model


    @property
    def vectorizer(self) -> Vectorizer:
        return self._vectorizer

    @vectorizer.setter
    def vectorizer(self, vectorizer: Vectorizer) -> None:
        self._model = vectorizer

    def get_data(self) -> None:
        ds_train = load_dataset("interpress_news_category_tr_lite",split="train")
        ds_test = load_dataset("interpress_news_category_tr_lite",split="test")
        print(ds_test)


    def preprocess(self) -> None:
        print("preprocess")

    def train(self) -> None:
        self._vectorizer.do_algorithm([])
        self._model.do_algorithm([])
        
        print("trained")
    def test(self) -> None:
        print("tested")