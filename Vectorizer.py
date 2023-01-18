from __future__ import annotations
from typing import List
from abc import abstractmethod

class Vectorizer:
    @abstractmethod
    def do_algorithm(self, data: List):
        pass


class TF_IDF(Vectorizer):
    def do_algorithm(self, data: List) -> List:
        print("tfidf")

class Word2Vec(Vectorizer):
    def do_algorithm(self, data: List) -> List:
        print("Word2Vec")

class Fasttext(Vectorizer):
    def do_algorithm(self, data: List) -> List:
        print("Fasttext")

class LanguageModel(Vectorizer):
    def do_algorithm(self, data: List) -> List:
        print("Language Model")