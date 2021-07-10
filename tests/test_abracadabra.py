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

sys.path.append(str(Path(__file__).parent.parent / 'src'))
from ppmg import PPMG, logger


directory = None


def write_to_files(ppmg):
    fn = directory / 'graph.dot'
    with open(fn, 'w') as ofile:
        print(nx.nx_agraph.to_agraph(ppmg.graph), file=ofile)
    sp.call(['dot', '-Tpdf', '-O', fn])
    fn = directory / 'ppmg.dot'
    with open(fn, 'w') as ofile:
        print(ppmg.to_dot(), file=ofile)
    sp.call(['dot', '-Tpdf', '-O', fn])


def test_example1():
    """Build a PPM-G model for the sequence "abracadabra"."""
    data = list("abracadabra")
    # We do not specify a graph, PPMG will assume an alphabet where all
    # possible sequences are allowed.
    ppmg = PPMG(depth=3)
    ppmg.learn_sequence(data)
    if directory:
        write_to_files(ppmg)

    print('P(x) = {}'.format(ppmg.probability('x', list())))
    print('P(d) = {}'.format(ppmg.probability('d', list())))
    print('P(d|ra) = {} (should be 0.07142)'.format(ppmg.probability('d', list('ra'))))
    print('P(b|ar) = {}'.format(ppmg.probability('b', list('ar'))))
    print('P(r|ab) = {}'.format(ppmg.probability('r', list('ab'))))
    print('P(r|abcdrrrabc) = {}'.format(ppmg.probability('d', list('abcdrrrabc'))))
    print('P(d|rabcdddab) = {}'.format(ppmg.probability('d', list('rabcdddab'))))
    print('P(d|dd) = {}'.format(ppmg.probability('d', list('dd'))))

    # print('P(.|ra) = {}'.format(ppmg.explore_probabilities(list('ra'))))
    print(ppmg.stats())


def test_example2():
    """Build a PPM-G model for the sequence "abracadabra"."""
    data = list("abracadabra")
    # We build the underlying graph data structure manually
    graph = nx.Graph()
    # All combinations are possible
    for b, e in combinations(set(data), 2):
        print('Add edge: {}--{}'.format(b, e))
        graph.add_edge(b, e)
    ppmg = PPMG(depth=3, graph=graph)
    ppmg.learn_sequence(data)
    if directory:
        write_to_files(ppmg)


def test_example3():
    """Build a PPM-G model for the sequence "abracadabra"."""
    data = list("abracadabra")
    # Only combinations that have been observed
    ppmg = PPMG(depth=3, autofill_graph='greedy')
    ppmg.learn_sequence(data)
    if directory:
        write_to_files(ppmg)

    print('P(d|ra) = {}'.format(ppmg.probability('d', list('ra'))))
    print('P(r|ab) = {}'.format(ppmg.probability('r', list('ab'))))
    print('Pr(x|a) = {}'.format(ppmg.probability('x', list('a'))))
    print('Pr(.|a) = {}'.format(ppmg.explore_probabilities(list('a'))))


def test_example4():
    """Build a PPM-G model for the sequence "abracadabra"."""
    data = list("abracadabra")
    # We do not specify a graph, PPMG will assume an alphabet where all
    # possible sequences are allowed.
    ppmg = PPMG(depth=3)
    ppmg.learn_sequence(data)
    if directory:
        write_to_files(ppmg)

    print('P(d|ra) = {}'.format(ppmg.probability('d', list('ra'))))
    print('P_avg(d|ra) = {}'.format(ppmg.probability_avg('d', list('ra'))))

    # print('P(.|ra) = {}'.format(ppmg.explore_probabilities(list('ra'))))
    print(ppmg.stats())


def test_example5():
    """Build a PPM-G model for the sequence "abracadabra"."""
    data = list("abracadabra")
    # We do not specify a graph, PPMG will assume an alphabet where all
    # possible sequences are allowed.
    ppmg = PPMG(depth=3)
    ppmg.learn_sequence(data)
    if directory:
        write_to_files(ppmg)

    print('l(arbcarab) = {}'.format(ppmg.logloss(list('arbcarab'))))
    print('l(abradabra) = {}'.format(ppmg.logloss(list('abradabra'))))
    print('l(abracadabra) = {}'.format(ppmg.logloss(list('abracadabra'))))


def test_example6():
    data = list("abracadabra")
    graph = nx.Graph()
    for b, e in combinations(set(data), 2):
        graph.add_edge(b, e)
    graph.add_edge("a","x")
    ppmg = PPMG(depth=3, graph=graph)
    ppmg.learn_sequence(data)

    print('Pr(x|a) = {}'.format(ppmg.probability('x', list('a'))))
    print('Pr(.|a) = {}'.format(ppmg.explore_probabilities(list('a'))))


if __name__ == '__main__':
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print(f"Saving files to {directory}")

    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging.WARNING)

    # test_example1()
    # test_example2()
    # test_example3()
    # test_example4()
    # test_example5()
    test_example6()
