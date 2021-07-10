#!/usr/bin/env python3
# encoding: utf-8
"""

Build a Prediction by Partial Match - Graph model.

Created by Wannes Meert.
Copyright (c) 2014-2021 KU Leuven. All rights reserved.
"""
import sys
import argparse
import logging
from pathlib import Path
from .ppmg import PPMG

logger = logging.getLogger("be.kuleuven.cs.dtai.ppmg")


def read_data(filenames, read_all=False, ignore=None):
    if not filenames:
        filenames = ['-']
    examples = []
    for filename in filenames:
        examples.append([])
        if filename == '-':
            print('Reading from STDIN')
            f = sys.stdin
        else:
            f = open(filename)
        try:
            for line in f:
                line = line.strip()
                if line.startswith('###') and not read_all:
                    break
                elif line.startswith('#'):
                    continue
                elif line:
                    line_parts = line.split()
                    event = line_parts[0]
                    if ignore is None or line not in ignore:
                        examples[-1].append(event)
                else:
                    examples.append([])
        except KeyboardInterrupt:
            print('\nKeyboard interrupt: stop reading')
        if filename != '-':
            f.close()
    return examples


def main(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Learn Prediction by Partial Match Graph model from data.',
        epilog='\n\nCopyright (c) 2013-2021 DTAI, KU Leuven.\nhttps://dtai.cs.kuleuven.be')
    parser.add_argument('--verbose', '-v', action='count', help="Verbose output")
    parser.add_argument('--depth', '-d', type=int, default=8, help="Tree depth")
    parser.add_argument('--autofill', default='greedy', help="Autofill type ('greedy', 'exhaustive')")
    parser.add_argument('--minamount', type=int, default=0, help="Minimum amount for edge to print")
    parser.add_argument('--stats', action='store_true', help='Print statistics about the learned model')
    parser.add_argument('--mergeapprox', type=int, help='Merge branches approximately if under the given amount')
    parser.add_argument('--mergelookahead', type=int, default=0, help='Lookahead while merging')
    parser.add_argument('--visualize', help='Save visualisations to this directory')
    parser.add_argument('data_files', nargs='*', help="Data files (STDIN if no files given)")

    # if argv is None:
    #     argv = sys.argv[1:]
    args = parser.parse_args(argv)

    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if args.verbose is None:
        logger.setLevel(logging.WARNING)
    elif args.verbose == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

    model = PPMG(depth=args.depth, autofill_graph=args.autofill)

    examples = read_data(args.data_files)
    for example in examples:
        model.learn_sequence(example)

    if args.stats:
        print(model.stats())

    if args.visualize:
        folder = Path(args.visualize)
        if not folder.exists():
            print('Folder does not exist: {}. Cannot save visualization.'.format(folder))
        else:
            with open(folder / 'ppmg.dot', 'w') as ofile:
                print('Writing output to {}'.format(ofile.name))
                print(model.to_dot(min_amount=args.minamount), file=ofile)
    else:
        folder = None

    if args.mergeapprox is not None:
        # noinspection PyUnusedLocal,PyUnusedLocal
        def subs_prob(x, y):
            return 0.1
        model.merge_approximate(min_amount=args.mergeapprox,
                                subs_prob=subs_prob,
                                lookahead=args.mergelookahead)

        if args.stats:
            print(model.stats())
        if args.visualize:
            with open(folder / 'ppmg_approx.dot', 'w') as ofile:
                print('Writing output to {}'.format(ofile.name))
                print(model.to_dot(), file=ofile)


# if __name__ == "__main__":
#     sys.exit(main())
