#! /usr/bin/env python3
"""
Example of how to use PPM-G.

Copyright (c) 2015 Wannes Meert, KU Leuven. All rights reserved.
This code is part of the PPM-G distribution.
"""
import os
import sys
import networkx as nx
from itertools import combinations
import logging
from pathlib import Path
import subprocess as sp
import pytest

sys.path.append(str(Path(__file__).parent.parent / 'src'))
from ppmg import PPMG, logger


directory = None


def test_example1():
    data = list("abracadabra")
    ppmg = PPMG(depth=3)
    ppmg.learn_sequence(data)
    # 1 / (5 + (4+2+1+1+1))
    assert ppmg.probability('d', list()) == pytest.approx(0.07142857142857142)
    assert ppmg.probability('d', list('ra')) == pytest.approx(0.07142857142857142)
    assert ppmg.probability('r', list('ab')) == pytest.approx(0.6666666666666666)


def test_example2():
    data = list("abracadabra")
    graph = nx.Graph()
    for b, e in combinations(set(data), 2):
        graph.add_edge(b, e)
    graph.add_edge("a","x")
    ppmg = PPMG(depth=3, graph=graph)
    ppmg.learn_sequence(data)

    assert ppmg.probability('d', list('ra')) == pytest.approx(0.07142857142857142)
    assert ppmg.probability('x', list('a')) == pytest.approx(0.04285714285714286)


def test_bug1():
    data = list("abracadabra")
    ppmg = PPMG(depth=3, autofill_graph="exhaustive")
    ppmg.learn_sequence(data)
    print(ppmg.probability('d', list('rabcdddab')))


if __name__ == '__main__':
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print(f"Saving files to {directory}")

    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging.WARNING)

    test_example1()
    test_example2()
    test_bug1()
