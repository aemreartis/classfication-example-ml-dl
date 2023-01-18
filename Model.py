from __future__ import annotations
from typing import List
from abc import abstractmethod

class Model:
    @abstractmethod
    def do_algorithm(self, data: List):
        pass


class LogisticRegression(Model):
    def do_algorithm(self, data: List) -> List:
        print("Logistic Regression")

class RandomForest(Model):
    def do_algorithm(self, data: List) -> List:
        print("Random Forest")

class SVM(Model):
    def do_algorithm(self, data: List) -> List:
        print("SVM")

class LinearSVM(Model):
    def do_algorithm(self, data: List) -> List:
        print("Linear SVM")

class XGBoost(Model):
    def do_algorithm(self, data: List) -> List:
        print("XGBoost")

