# -*- coding: UTF-8 -*-
"""
ppmg.ppmg
~~~~~~~~~

Prediction by Partial Match - Graph

:author: Wannes Meert
:copyright: Copyright 2014-2021 KU Leuven, DTAI Research Group.
:license: MIT License, see LICENSE for details.
"""
import operator
import json
import re
import logging
import sys
import math
from collections import defaultdict, namedtuple

logger = logging.getLogger("be.kuleuven.cs.dtai.ppmg")

try:
    from tqdm import tqdm

    class TqdmLoggingHandler(logging.Handler):
        def __init__(self, level=logging.NOTSET):
            super(self.__class__, self).__init__(level)

        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                self.handleError(record)
    streamHandler = TqdmLoggingHandler()
except ImportError:
    # noinspection PyPep8Naming
    class tqdm:
        def __init__(self, a):
            self.a = a

        def __iter__(self):
            return self.a.__iter__()

        def update(self, i=1):
            pass

    streamHandler = logging.StreamHandler(sys.stdout)

# Maximal number of meta information to store in a node.
# For nodes close to the root, this list may become too large
meta_max_length = 10000
# Only store meta information for nodes deeper than the given value
meta_min_depth = 4


MergeResult = namedtuple('MergeResult', 'left right')


def apply_meta(meta, func):
    if meta is None:
        return
    if type(meta) == MergeResult:
        apply_meta(meta.left, func)
        apply_meta(meta.right, func)
        return
    for pos, depth, meta_info in meta:
        func(pos, depth, meta_info)


class PPMG:
    """Prediction by Partial Match - Graph is a variation of the PPM-C algorithm [1]
    that assumes a graph underneath instead of an alphabet. This takes into
    account that in a graph not every node is reachable from every other node. This
    is what the PPM-C algorithm does to compute an escape but in environments with
    large branching factors the number of seen pairs is potentially negligible
    compared to the not observed pairs.

    The graph expresses a constraints on the possible sequences that are expected
    to exist (it is not enforced in the current version, but the escape is not
    well defined in such cases).

    [1] Ron Begleiter, Ran El-Yaniv, and Golan Yona. "On Prediction Using Variable
        Order Markov Models". In: Journal of Artificial Intelligence Research 22
        (2004), pp. 385-421.
    """
    FILL_EXHAUSTIVE = 'exhaustive'
    FILL_GREEDY = 'greedy'

    def __init__(self, depth=4, graph=None, autofill_graph=None, learn_until_end=False):
        """Create a PPM-G data structure.
        :param depth:
        :param graph: Compute escape values based on the given graph.
            Should be compatible with a NetworkX graph.
        :param autofill_graph: Method to extend graph while learning
               (None, 'none', 'exhaustive', 'greedy')
        :return:
        """
        self.root = TreeNode('^', 0)
        self.depth = depth
        self.autofill_graph = None
        if autofill_graph == 'exhaustive':
            self.autofill_graph = self.FILL_EXHAUSTIVE
        elif autofill_graph == 'greedy':
            self.autofill_graph = self.FILL_GREEDY
        if graph is None:
            try:
                import networkx as nx
            except ImportError:
                raise ImportError('If no graph is given, the networkx package is required.')
            self.graph = nx.Graph()
            if self.autofill_graph is None:
                self.autofill_graph = self.FILL_EXHAUSTIVE
        else:
            self.graph = graph

        # Learn also last items in sequence that are closer than depth to end.
        # This is useful if every possible node should also be available in
        # the root level
        self.learn_until_end = learn_until_end

    def extract_kbest(self, k=10):
        """Generate a new model with only the k best paths."""
        raise Exception('TODO')

    def stats(self):
        """Return stats about the model."""
        stats = 'Stats:\n'
        node_stats = {
            'size': 0,
            'maxweight': 0,
            'minweight': 0,
            'avgweight': 0,
            'endpoints': 0,
            'maxdepth': 0,
            'avgdepth': 0,
            'mindepth': 0
        }

        def compute_stats(node):
            node_stats['size'] += 1
            node_stats['avgweight'] += node.amount
            if node_stats['maxweight'] < node.amount:
                node_stats['maxweight'] = node.amount
            if node_stats['minweight'] > node.amount:
                node_stats['minweight'] = node.amount
            if not node.has_children():
                node_stats['avgdepth'] += node.depth
                if node_stats['maxdepth'] < node.depth:
                    node_stats['maxdepth'] = node.depth
                if node_stats['mindepth'] > node.depth:
                    node_stats['mindepth'] = node.depth
                node_stats['endpoints'] += 1
        self.root.node_visitor(compute_stats)
        node_stats['avgweight'] /= node_stats['size']
        node_stats['avgdepth'] /= node_stats['size']
        stats += 'Alphabet size:   {}\n'.format(len(self.graph.nodes()))
        stats += 'Number of nodes: {}\n'.format(node_stats['size'])
        stats += 'Average weight:  {}\n'.format(node_stats['avgweight'])
        stats += 'Maximal weight:  {}\n'.format(node_stats['maxweight'])
        stats += 'Minimal weight:  {}\n'.format(node_stats['minweight'])
        stats += 'Average depth:   {}\n'.format(node_stats['avgdepth'])
        stats += 'Maximal depth:   {}\n'.format(node_stats['maxdepth'])
        stats += 'Minimal depth:   {}\n'.format(node_stats['mindepth'])
        stats += 'Sequences:       {}\n'.format(node_stats['endpoints'])
        return stats

    def add_pairwise_edges(self, sequence):
        """Add edges for all transitions in given sequence.
        :param sequence: List of symbols
        """
        for b, e in zip(sequence[:-1], sequence[1:]):
            self.graph.add_edge(b, e)

    def add_exhaustive_edges(self, symbols):
        """Add all possibles edges between given letters.
        :param symbols: Set of symbols
        """
        alphabet = set(symbols)
        alphabet.difference_update(self.graph.nodes())
        alphabet = list(alphabet)
        logger.debug('Automatically adding letters: {}'.format(alphabet))
        if len(self.graph.nodes()) == 0:
            self.graph.add_node(alphabet[0])
        for letter in alphabet:
            for other in alphabet:  # self.graph.nodes():
                # if letter != other:
                logger.debug('Add edge {}--{}'.format(letter, other))
                self.graph.add_edge(letter, other)

    def learn_sequence(self, sequence, show_progress=True, meta=None, condition=None):
        """Incrementally learn a variable order n-gram from the given sequence.
        This method should be called for all sequences available, the results
        are incrementally merged into the current model.
        :param sequence: The sequence to process.
        :param show_progress: Print progress while processing substrings.
        :param meta: Meta-data that will be stored in the final nodes of the
                     tree.
        :param condition: Function that takes an array and returns boolean.
        """
        if self.autofill_graph == self.FILL_EXHAUSTIVE:
            self.add_exhaustive_edges(sequence)
        elif self.autofill_graph == self.FILL_GREEDY:
            self.add_pairwise_edges(sequence)

        if self.learn_until_end:
            # sequence += self.depth*[-1]
            r = range(len(sequence))
        else:
            r = range(len(sequence)-self.depth+1)
        if len(sequence) > 100 and show_progress:
            tqdm(r)
        for start in r:
            if condition is None or condition(sequence[start:]):
                self.root.add_sequence(sequence[start:], self.depth,
                                       pos=start, meta=meta)

    def probability(self, symbol, context):
        """P(symbol | context)
        :param symbol:
        :param context: List of history/context symbols
        :return: Probability
        """
        logger.debug("probability({},{})".format(symbol, context))
        return self.root.calculate_global_chance(context[-self.depth + 1:], symbol, self.graph)

    def probability_avg(self, symbol, context):
        """E[P(symbol | context[i:])]
        Average probability over all possible subsets of the context.
        :param symbol:
        :param context: List of history/context symbols
        :return: Probability
        """
        logger.debug("probability_avg({},{})".format(symbol, context))
        pr = []
        for i in range(len(context)):
            pr.append(self.probability(symbol, context[i:]))
        return sum(pr)/len(context)

    def explore_probabilities(self, context, normalize=False):
        """Iterate over all possible next symbols s and return P(s | context)
        """
        return self.root.explore_chances(context, self.graph, self.root, normalize)

    def logloss(self, symbols):
        """Compute the average log-loss of the given sequence.

        l(P,x^T_1) = -1/T*sum(log(P(x_i|x_1, ..., x_{i-1})))

        :param symbols: List of symbols
        """
        t = len(symbols)
        l = 0
        for i in range(t):
            l += math.log(self.probability(symbols[i], symbols[:i]), 2)
        return -1/t*l

    def sequences(self, min_amount=0, min_length=None, only_endpoints=False,
                  min_diff=0):
        """Iterate over all possible sequences.
        :param min_amount: Only return sequences that appear at least this
            amount of times.
        :param min_length: Only return sequences of this length or longer.
        :param only_endpoints: Only return sequences that are leafs of the tree
            or of which the subtrees would not return any sequences given the
            other criteria (min_amount and min_length).
        :param min_diff: If not only endpoints, only return paths up to nodes
            that are at least this amount larger than any of their children.
        """
        return Sequences(self.root.sequences([], 0,
                                             min_amount=min_amount,
                                             min_length=min_length,
                                             only_endpoints=only_endpoints,
                                             min_diff=min_diff))

    def drop_ifall(self, condition):
        """Drop all branches where a sequence contains only nodes for which
        the given condition is true."""
        self.root.drop_ifall(condition)

    def merge_approximate(self, min_amount=0, subs_prob=None, min_prob=0.0, lookahead=0):
        """Merge branches into the most likely other branch if they do not
        occur an expected minimal amount.
        :param min_amount: Try to merge if branch has lower than this amount.
        :param subs_prob: Function subs_prob(x,y) that expresses the
                          probability that x can be replaced by y or
                          Pr(value=y | value=x).
        :param min_prob:
        :param lookahead: If the lookahead is larger than zero, the Jaccard
                          measure is recursively computed to compute the
                          similarity of the subtrees. The resulting value is
                          combined with the substitution probability.
        """
        q = [(self.root, set())]  # LIFO queue of (next node, keep children)
        while len(q) != 0:
            node, keep_children = q.pop()
            logger.debug('Try to merge children of {} -- keep [{}]'.format(
                node,
                ', '.join([str(c) for c in keep_children])))
            if node.children is None or len(node.children) == 0:
                logger.debug('No children')
                continue
            if len(node.children) == 1:
                logger.debug('Only one child')
                q.append((node.children[0], set()))
                continue
            min_node, min_node_amount = None, 9999999
            total_amount = 0.0
            for child in node.children:
                total_amount += child.amount
                if child not in keep_children and \
                   child.amount < min_amount and \
                   child.amount < min_node_amount:
                    min_node, min_node_amount = child, child.amount
            if min_node is None:
                logger.debug('No min node to merge')
                q.extend([(c, set()) for c in node.children])
                continue
            logger.debug('Min node:    {}'.format(min_node))
            target_node, target_ll = None, min_prob
            for child in node.children:
                if child != min_node:
                    cur_ll = child.amount/total_amount*subs_prob(min_node.name, child.name)
                    if lookahead > 0 and len(child.children) != 0 and len(min_node.children) != 0:
                        jaccard = self.jaccard_avg(child, min_node, lookahead)
                        logger.debug('Match node:  {} = {:.4f}*{:.4f} = {:.4f}'.format(
                            child, cur_ll, jaccard, cur_ll*jaccard))
                        cur_ll *= jaccard
                    else:
                        logger.debug('Match node:  {} = {:.4f}'.format(child, cur_ll))
                    if cur_ll > 0 and cur_ll > target_ll:
                        target_node, target_ll = child, cur_ll
            if target_node is None:
                logger.debug('No target node found to merge')
                keep_children.add(min_node)
                q.append((node, keep_children))
                continue
            logger.debug('Target node: {} (prob={:.4f})'.format(target_node, target_ll))
            # print('{:<5} Match found {}({}) & {}({}): {:.4f}'.format(
            #   target_node.hash, min_node, min_node.hash, target_node, target_node.hash, target_ll))
            node.children.remove(min_node)
            target_node.merge(min_node)
            q.append((node, keep_children))

    def jaccard_avg(self, this, other, lookahead=0):
        """How similar are the two subtrees.

        - The Jaccard measure is computed between the children of both subtrees.
        - The avarage Jaccard measure for the children is computed for all
          children that appear in both subtrees if the lookahead limit is not
          reached.
        - The two obtained Jaccard measures are averaged.
        """
        if len(this.children) == 0 and len(other.children) == 0:
            return 1
        this_names = set([c.name for c in this.children])
        other_names = set([c.name for c in other.children])
        all_names = this_names | other_names
        both_names = this_names & other_names
        jaccard = len(both_names) / len(all_names)
        if lookahead == 0:
            logger.debug('Jaccard between {} and {}: {:.4f} (lookahead = 0)'.format(this, other, jaccard))
            return jaccard
        if len(both_names) == 0:  # No overlap
            # return 0.0
            # res =  1/(1+math.exp(6.0-10.0*(len(self.graph.nodes())-len(all_names))/len(self.graph.nodes())))
            res = (1.0*len(self.graph.nodes())-len(all_names))/len(self.graph.nodes())
            res /= lookahead
            logger.debug('Jaccard between {} and {}: {:.4f} (lookahead = {}, no overlap)'.format(
                this, other, jaccard, lookahead))
            return res
            # return 1.0/len(all_names)

        both_children = defaultdict(lambda: [None, None])
        jaccard_children = 0.0
        for child in this.children:
            if child.name in both_names:
                both_children[child.name][0] = child
        for child in other.children:
            if child.name in both_names:
                both_children[child.name][1] = child
        for c in both_children.values():
            jaccard_children += self.jaccard_avg(c[0], c[1], lookahead-1)
        jaccard_children /= len(both_children)
        jaccard2 = (jaccard+jaccard_children)/2
        logger.debug('Jaccard between {} and {}: ({:.4f}+{:.4f})/2 = {:.4f} (lookahead = {})'.format(
            this, other, jaccard, jaccard_children, jaccard2, lookahead))
        return jaccard2

    def to_dot(self, min_amount=0):
        """Export tree structure to Graphviz dot format.
        :param min_amount: Ignore branches with an amount lower than this.
        """
        dot = '  digraph ppmg {\nrankdir=LR;\n'
        dot += self.root.to_dot(min_amount)
        dot += '}\n'
        return dot

    def to_ascii(self):
        r = ''
        if self.root is not None:
            r += '{}\n'.format(self.root.name)
            r += self.root.to_ascii()
        return r

    def to_json(self):
        nodes = self.root.to_list()
        return json.dumps(nodes, indent=2)

    @staticmethod
    def from_json(data):
        nodes = json.loads(data)
        nodes_dict = dict()
        for hsh, name, amount, depth, parent in nodes:
            node = TreeNode(name, amount, depth)
            node.hash = hsh
            nodes_dict[hsh] = node
            nodes_dict[parent].add_child(node)

    def __str__(self):
        if self.root is None:
            return ''
        else:
            return self.to_ascii()


num_nodes = 0
dot_ignored_chars = re.compile("""["]""")


class TreeNode(object):
    """A PPMG tree consists of TreeNodes.
    """

    __slots__ = ('name', 'amount', 'depth', 'hash', 'meta', 'children')

    def __init__(self, name, amount=0, depth=0, children=None):
        global num_nodes
        self.name = name
        self.amount = amount
        self.depth = depth
        num_nodes += 1
        self.hash = num_nodes
        self.meta = None
        if children is None:
            self.children = dict()
        else:
            self.children = children

    def has_children(self):
        return self.children is not None and len(self.children) > 0

    def to_ascii(self, level=0):
        """Export as ASCII format."""
        txt = ''
        indent = '  '*level
        for child in self.children.values():
            txt += '{}+-{} ({})\n'.format(indent, child.name, self.amount)
            txt += child.to_ascii(level + 1)
        return txt

    def to_dot(self, min_amount=0):
        """Export as Graphviz dot format."""
        clean_name = dot_ignored_chars.sub("", str(self.name))
        dot = '  {} [label="{}"];\n'.format(self.hash, clean_name)
        for child in self.children.values():
            if child.amount >= min_amount:
                dot += '  {} -> {} [label="{}"];\n'.format(self.hash, child.hash, child.amount)
                dot += child.to_dot(min_amount)
        return dot

    def to_list(self, parent=None):
        hsh = parent.hash if parent is None else ""
        nodes = [(self.hash, self.name, self.amount, self.depth, hsh)]
        for child in self.children.values():
            nodes.extend(child.to_list(self))
        return nodes

    def __str__(self):
        return '{}-{}[{}]'.format(self.depth, self.name, self.amount)

    def node_visitor(self, func):
        func(self)
        for child in self.children.values():
            child.node_visitor(func)

    def add_child(self, child):
        """
        :param child: Node
        :return: None
        """
        self.children[child.name] = child

    def add_meta(self, pos, depth, meta):
        if meta is None or depth < meta_min_depth or depth == 0:
            return
        if self.meta is None:
            self.meta = [(pos, depth, meta)]
        elif len(self.meta) >= meta_max_length:
            return
        else:
            self.meta.append((pos, depth, meta))

    def add_sequence(self, sequence, max_depth=0, pos=0, meta=None):
        """
        Learn a new sequence starting from this node.
        :param sequence:
        :param max_depth:
        :param pos:
        :param meta: Meta-data that will be stored in the final nodes of the
                     tree.
        :return: None
        """
        # TODO: Add option to only expand a branch if a second entry appears.
        #       This will require that a pattern of length k appears
        #       k+l times to have a count of 1+l.
        if self.depth >= max_depth or len(sequence) == 0:
            return
        for node in self.children.values():
            if node.name == sequence[0]:
                # Add to existing child
                node.amount += 1
                node.add_meta(pos+1, self.depth+1, meta)
                if len(sequence) > 1:
                    node.add_sequence(sequence[1:], max_depth, pos+1, meta=meta)
                return
        # Add a new child
        child = TreeNode(sequence[0], 1, self.depth+1)
        self.add_child(child)
        child.add_meta(pos+1, self.depth+1, meta)
        child.add_sequence(sequence[1:], max_depth, pos+1, meta=meta)

    def merge(self, other):
        """Merge the subtree of the given other node with this subtree.
        :param other: A TreeNode object
        :return: None
        """
        logger.debug('Merging {} and {} -- amount = {}+{} = {}'.format(
            self, other, self.amount, other.amount, self.amount+other.amount))
        self.amount += other.amount
        if other.meta is not None:
            if self.meta is None:
                self.meta = []
            # self.meta.extend(other.meta)
            self.meta = MergeResult(left=self.meta, right=other.meta)
        if other.children is None:
            return
        if self.children is None:
            self.children = list()
        for other_child in other.children:
            found = False
            if self.children is not None:
                for child in self.children.values():
                    if other_child.name == child.name:
                        child.merge(other_child)
                        found = True
                        break
            if not found:
                logger.debug('Child is new, append')
                self.children.append(other_child)

    def calculate_global_chance(self, context, symbol, graph):
        logger.debug("calculate_global_chance({}, {}, {})".format(self, context, symbol))
        return self.calculate_chance(context, symbol, context, self, graph)

    def calculate_chance(self, context, symbol, original_context, tree, graph):
        """Check whether the context \sigma exists in the tree. If it exists
        compute the transfer probability to the given symbol, otherwise
        apply the escape mechanism and retry on a shorter context.

        :param context: The remainder of the context we are searching for
            in the tree.
        :param symbol: The query
        :param original_context: The context we use to define the transition
            probability (s^n_{n-D+1})
        :param tree: The root of the tree
        :param graph: A graph expressing the possible transitions
        """
        logger.debug("calculate_chance({}, {}, {}, {}, {})".format(self, context, symbol, original_context, tree))
        found = False
        result = None
        if len(context) == 0:
            result = self.check_chance(symbol, original_context, tree, graph)
        else:
            if context[0] in self.children:
                found = True
                child = self.children[context[0]]
                if len(context) > 1:
                    result = child.calculate_chance(context[1:], symbol, original_context, tree, graph)
                else:
                    result = child.check_chance(symbol, original_context, tree, graph)
            if not found:
                if len(original_context) > 1:
                    result = tree.calculate_global_chance(original_context[1:], symbol, graph)
                else:
                    result = float(1)/float(graph.degree(context[0]))
        logger.debug("calculate_chance = {}".format(result))
        return result

    def check_chance(self, symbol, original_context, tree, graph):
        """Compute the probability of the symbol from the current node that
        represent the last symbol in the given context.
        """
        logger.debug("check_chance({}, {}, {}, {})".format(self, symbol, original_context, tree))
        amount_of_symbol = 0
        total_amount = 0
        found = False
        result = None
        for child in self.children.values():
            if child.name == symbol:
                found = True
                amount_of_symbol = child.amount
            total_amount = total_amount + child.amount
        if found:  # \sigma \in \Sigma_s
            # P_k(\sigma | s) = N(s\sigma)/(|\Sigma_s| + sum(N(s\sigma')))
            result = float(amount_of_symbol)/float(len(self.children) + total_amount)
        else:  # escape and recurse
            if len(original_context) == 0:
                # No context, can be any node in the graph
                if self.name in graph:
                    # Symbol appears in graph but in no training data
                    result = 1.0/(len(graph.nodes()) + total_amount)
                else:
                    # Symbol does not appear in graph or training data
                    result = 0.0
            else:
                # P_k(escape| s) = |\Sigma_s|/(|\Sigma_s| + sum(N(s\sigma')))
                p_escape = (float(len(self.children))/float(len(self.children) + total_amount))
                if len(original_context) > 1:
                    # s^n_{n-D+1}\sigma is not in training sequence, escape and restart at s^n_{n-D+2}
                    result = p_escape * tree.calculate_global_chance(original_context[1:], symbol, graph)
                else:
                    # s^n_{n-D+1}\sigma is not in training sequence and the next context is the
                    # empty context.
                    # PPM-C would use P(\sigma|\epsilon) = 1/|\Sigma|, we use 1/sum(weight(neigbour)+1) for
                    # the current symbol instead of 1/alphabet over the entire graph.
                    total_amount = 0
                    graph_neighbors = list(graph.neighbors(self.name))
                    for neighbor in graph_neighbors:
                        if neighbor in tree.children:
                            total_amount += tree.children[neighbor].amount
                    logger.debug("total_amount: {}".format(total_amount))
                    logger.debug("check_chance (multiply with graph degree: {})".format(
                        1.0/(len(graph_neighbors)+total_amount)))
                    result = p_escape * 1.0/(len(graph_neighbors)+total_amount)
                logger.debug("check_chance (multiply with escape: {})".format(p_escape))
        logger.debug("check_chance = {}".format(result))
        return result

    @staticmethod
    def explore_chances(context, graph, tree, normalize=False):
        last = context[len(context)-1]
        # print('last = {}'.format(last))
        neighbors = graph.neighbors(last)
        next_nodes = list()
        absolute_chances = list()
        normalized_chances = dict()
        total = 0
        for nbr in neighbors:
            result = tree.calculate_global_chance(context, nbr, graph)
            absolute_chances.append(result)
            total += result
            next_nodes.append(nbr)
        count = 0
        # print(absolute_chances)
        for nbr in neighbors:
            if normalize:
                normalized_chances[nbr] = float(absolute_chances[count])/float(total)
            else:
                normalized_chances[nbr] = float(absolute_chances[count])
            count += 1
        return normalized_chances

    def multiple_explore(self, context, amount, graph, tree):
        print('context: ')
        print(context)
        next_nodes = self.explore_chances(context, graph, tree)
        print(next_nodes)
        sorted_next_nodes = sorted(next_nodes.items(), key=operator.itemgetter(1), reverse=True)
        amount_chosen = len(sorted_next_nodes)
        chosen_nodes = list()
        chosen_nodes.append((context[-1], 1, context[-1]))
        for node in sorted_next_nodes:
            chosen_nodes.append((node[0], node[1], context[-1]))
        final_result = list()
        for node in sorted_next_nodes:
            final_result.append((node[0], node[1], context))
        index = 0
        while amount_chosen < amount:
            (next_node, next_chance, next_context) = final_result[index]
            print('next_context 1: ')
            print(next_context)
            next_context_clone = list()
            for n in next_context:
                next_context_clone.append(n)
            next_context_clone.append(next_node),
            print('next_context 2: ')
            print(next_context_clone)
            search_context = next_context_clone[-(d-1):]
            next_nodes = self.explore_extended_chances(search_context, graph, next_chance, tree)
            print(next_nodes)
            for node in next_nodes.items():
                final_result.append((node[0], node[1], next_context_clone))
            final_result = sorted(final_result, key=operator.itemgetter(1), reverse=True)
            index += 1
            (next_node, next_chance, next_context) = final_result[index]
            # print final_result
            found = False
            for node in chosen_nodes:
                # changed to take last node into account for direction
                if next_node == node[0] and next_context[-1] == node[3]:
                    found = True
            if not found:
                chosen_nodes.append((next_node, next_chance, next_context[-1]))
                amount_chosen += 1
        coords = dict()
        index = 0
        for node in chosen_nodes:
            coords[index] = (node[0], node[1], node[2], node[3])
            index += 1
        return coords

    @staticmethod
    def explore_extended_chances(context, graph, chance, tree):
        neighbors = graph.neighbors(context[len(context)-1])
        next_nodes = list()
        absolute_chances = list()
        normalized_chances = dict()
        total = 0
        for nbr in neighbors:
            result = tree.calculate_global_chance(context, nbr, graph)
            absolute_chances.append(result)
            total += result
            next_nodes.append(nbr)
        count = 0
        for nbr in neighbors:
            # print 'chance for ' + str(nbr)
            normalized_chances[nbr] = (float(absolute_chances[count]) / float(total)) * chance
            # print normalizedChances[nbr]
            count += 1

        return normalized_chances

    def sequences(self, sequence, length, min_amount=0, min_length=None,
                  only_endpoints=False, min_diff=0):
        """Iterate over all possible sequences.
        :param min_diff: Minimal difference between current node and the only
                         child to return both sequences
        """
        if self.children is None or len(self.children) == 0:
            if min_length is None or len(sequence) >= min_length:
                return [(sequence + [self.name], self.amount, self.meta)]
            else:
                return []
        else:
            result = []
            cur_min_diff = self.amount
            max_amount = 0
            for child in self.children.values():
                if child.amount >= min_amount:
                    cur_min_diff = min(cur_min_diff, self.amount-child.amount)
                    max_amount = max(max_amount, child.amount)
                    result.extend(child.sequences(sequence + [self.name],
                                                  length + 1,
                                                  min_amount=min_amount,
                                                  min_length=min_length,
                                                  only_endpoints=only_endpoints,
                                                  min_diff=min_diff))
            if only_endpoints:
                if len(result) == 0 and (min_length is None or len(sequence) >= min_length):
                    # Also include nodes of which the children will not produce
                    # any sequences because they do not fullfil the condittions.
                    result.append((sequence + [self.name], self.amount, self.meta))
            else:
                if min_length is None or len(sequence) >= min_length:
                    if cur_min_diff >= min_diff:
                        result.append((sequence + [self.name], self.amount, self.meta))
            return result

    def drop_ifall(self, condition):
        new_children = []
        for child in self.children.values():
            if not child.drop_ifall(condition):
                new_children.append(child)
        self.children = new_children
        if len(self.children) == 0 and condition(self):
            return True
        return False


class Sequences(list):
    """List of sequences.
    Format: list((<count>, <pattern>, <meta>))
    """
    def __init__(self, *args, **kwargs):
        """Create a set of sequences.

        Potential arguments:
        :param dist_fun: Use this function to compute the distance between two
            sequences (default: slow Python dtw implementation).
        :param dists_func: Use this function to compute the distance matrix
            between all given sequences (default: serial Python loop).
        """
        self.dist = None
        if 'dist_fun' in kwargs:
            self.dist = kwargs['dist_fun']
            del kwargs['dist_fun']
        if self.dist is None:
            self.dist = self.dtw_distance
        self.dists = None
        if 'dists_fun' in kwargs:
            self.dists = kwargs['dists_fun']
            del kwargs['dists_fun']
        if self.dists is None:
            self.dists = self.dtw_distances
        super(Sequences, self).__init__(*args, **kwargs)

    def filter(self, condition):
        return Sequences((t for t in self if condition(t)))

    def merge(self, max_dist=0.005, merge_rest_func=None,
              max_length_diff=5,
              window=None, max_step=None, penalty=None):
        """Merge sequences.
        """
        import numpy as np
        dist_opts = {
            'max_dist': max_dist,
            'max_step': max_step,
            'window': window,
            'max_length_diff': max_length_diff,
            'penalty': penalty
        }
        if max_length_diff is None:
            max_length_diff = np.inf
        cur_seqs = [p[0] for p in self]  # list(number)
        cur_meta = [p[1:3] for p in self]  # list((cnt, merge))
        cur_rest = [p[3:] for p in self]  # list(*rest)
        large_value = np.inf
        logger.info('Computing distances')
        dists = self.dists(cur_seqs, **dist_opts)
        # min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        # min_value = dists[min_idx]
        min_value = np.min(dists)
        min_idxs = np.argwhere(dists == min_value)
        min_idx = -1
        max_cnt = 0
        for r, c in [min_idxs[ii, :] for ii in range(min_idxs.shape[0])]:
            total = cur_meta[r][0] + cur_meta[c][0]
            if total > max_cnt:
                max_cnt = total
                min_idx = (r, c)
        deleted = set()
        cnt_merge = 0
        logger.info('Merging patterns')
        pbar = tqdm(total=dists.shape[0])
        # Hierarchical clustering (distance to prototype)
        while min_value <= max_dist:
            cnt_merge += 1
            i1, i2 = min_idx[0], min_idx[1]
            p1 = cur_seqs[i1]
            c1, m1 = cur_meta[i1]
            p2 = cur_seqs[i2]
            c2, m2 = cur_meta[i2]
            if c1 < c2 or (c1 == c2 and len(p1) > len(p2)):
                i1, c1, m1, i2, c2, m2 = i2, c2, m2, i1, c1, m1
            logger.debug("Merge {} <- {} ({:.3f})".format(i1, i2, min_value))
            cur_meta[i1] = (c1 + c2, MergeResult(left=m1, right=m2))
            if merge_rest_func is not None:
                cur_rest[i1] = merge_rest_func(cur_rest[i1], cur_rest[i2])
            else:
                cur_rest[i1] = []
            # if recompute:
            #     for r in range(i1):
            #         if r not in deleted and abs(len(cur_seqs[r]) - len(cur_seqs[i1])) <= max_length_diff:
            #             dists[r, i1] = self.dist(cur_seqs[r], cur_seqs[i1], **dist_opts)
            #     for c in range(i1+1, len(cur_seqs)):
            #         if c not in deleted and abs(len(cur_seqs[i1]) - len(cur_seqs[c])) <= max_length_diff:
            #             dists[i1, c] = self.dist(cur_seqs[i1], cur_seqs[c], **dist_opts)
            for r in range(i2):
                dists[r, i2] = large_value
            for c in range(i2 + 1, len(cur_seqs)):
                dists[i2, c] = large_value
            deleted.add(i2)
            pbar.update(1)
            # min_idx = np.unravel_index(np.argmin(dists), dists.shape)
            # min_value = dists[min_idx]
            min_value = np.min(dists)
            min_idxs = np.argwhere(dists == min_value)
            min_idx = -1
            max_cnt = 0
            for r, c in [min_idxs[ii, :] for ii in range(min_idxs.shape[0])]:
                total = cur_meta[r][0] + cur_meta[c][0]
                if total > max_cnt:
                    max_cnt = total
                    min_idx = (r, c)

        new_sequences = Sequences()
        for i, (s, m, r) in enumerate(zip(cur_seqs, cur_meta, cur_rest)):
            if i not in deleted:
                new_sequences.append(tuple([s, m[0], m[1]] + list(r)))
        return new_sequences

    # noinspection PyUnusedLocal
    @staticmethod
    def dtw_distance(s1, s2,
                     window=None, max_dist=None,
                     max_step=None, max_length_diff=None, penalty=None):
        """
        Dynamic Time Warping
        NOTE: This is a slow method. Use the dtaidistance library for faster methods
        :param s1: First sequence of numbers
        :param s2: Second sequence of numbers
        :param window: Only allow for shifts up to this amount away from the two diagonals
        :param max_dist: Stop if the returned values will be larger than this value
        :param max_step: Do not allow steps larger than this value
        :param max_length_diff: Sequences can differ at most this amount
        :param penalty: Penalty to add if compression or expansion is applied

        Returns: DTW distance
        """
        import numpy as np
        r, c = len(s1), len(s2)
        if max_length_diff is not None and abs(r - c) > max_length_diff:
            return np.inf
        if window is None:
            window = max(r, c)
        if max_step is None:
            max_step = np.inf
        if max_dist is None:
            max_dist = np.inf
        length = min(c + 1, abs(r - c) + 2 * (window - 1) + 1 + 1 + 1)
        # print("length (py) = {}".format(length))
        dtw = np.full((2, length), np.inf)
        dtw[0, 0] = 0
        last_under_max_dist = 0
        skip = 0
        i0 = 1
        i1 = 0
        for i in range(r):
            # print("i={}".format(i))
            # print(dtw)
            if last_under_max_dist == -1:
                prev_last_under_max_dist = np.inf
            else:
                prev_last_under_max_dist = last_under_max_dist
            last_under_max_dist = -1
            skipp = skip
            skip = max(0, i - max(0, r - c) - window + 1)
            i0 = 1 - i0
            i1 = 1 - i1
            dtw[i1, :] = np.inf
            if dtw.shape[1] == c + 1:
                skip = 0
            for j in range(max(0, i - max(0, r - c) - window + 1), min(c, i + max(0, c - r) + window)):
                d = (s1[i] - s2[j]) ** 2
                if d > max_step:
                    continue
                assert j + 1 - skip >= 0
                assert j - skipp >= 0
                assert j + 1 - skipp >= 0
                assert j - skip >= 0
                dtw[i1, j + 1 - skip] = d + min(dtw[i0, j - skipp],
                                                dtw[i0, j + 1 - skipp] + penalty,
                                                dtw[i1, j - skip] + penalty)
                # print('({},{}), ({},{}), ({},{})'.format(i0, j - skipp, i0, j + 1 - skipp, i1, j - skip))
                # print('{}, {}, {}'.format(dtw[i0, j - skipp], dtw[i0, j + 1 - skipp], dtw[i1, j - skip]))
                # print('i={}, j={}, d={}, skip={}, skipp={}'.format(i,j,d,skip,skipp))
                # print(dtw)
                if dtw[i1, j + 1 - skip] <= max_dist:
                    last_under_max_dist = j
                else:
                    # print('above max_dist', dtw[i1, j + 1 - skip], i1, j + 1 - skip)
                    dtw[i1, j + 1 - skip] = np.inf
                    if prev_last_under_max_dist + 1 - skipp < j + 1 - skip:
                        # print("break")
                        break
            if last_under_max_dist == -1:
                # print('early stop')
                # print(dtw)
                return np.inf
        return dtw[i1, min(c, c + window - 1) - skip]

    def dtw_distances(self, seqs, *args, **kwargs):
        """Compute distance matrix between als sequences of numbers.
        :param seqs: Sequence of sequence of numbers
        :param args: -
        :param kwargs: -
        :return: Numpy matrix of distances (only upper triangle is filled)
        """
        import numpy as np
        logger.info("Compute distances")
        dists = np.zeros((len(seqs), len(seqs))) + np.inf
        for r in tqdm(range(len(seqs))):
            for c in range(r + 1, len(seqs)):
                dists[r, c] = self.dist(seqs[r], seqs[c], *args, **kwargs)
        return dists
