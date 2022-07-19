import pandas as pd
from pytest import fixture
from src.pairs import Pairs


@fixture
def pairs() -> Pairs:
    return Pairs(pd.read_csv('..\\..\\data\\test.csv'))

def test_pairs_number(pairs: Pairs):
    assert pairs.number == 8
