# Prediction by Partial Match - Graph

Prediction by Partial Match - Graph is a variation of the PPM-C algorithm [1-3]
that assumes a graph underneath instead of an alphabet. This takes into
account that in a graph not every node is reachable from every other node. This
is what the PPM-C algorithm does to compute an escape but in environments with
large branching factors the number of seen pairs is potentially negligible
compared to the not observed pairs.

The graph expresses a constraints on the possible sequences that are expected
to exist (it is not enforced in the current version, but the escape is not
well defined in such cases).

## Example

    data = list("abracadabra")
    # We do not specify a graph, PPMG will assume an alphabet where all
    # possible sequences are allowed.
    ppmg = PPMG(depth=3)
    ppmg.learn_sequence(data)

    print('P(x) = {}'.format(ppmg.probability('x', list())))
    print('P(d) = {}'.format(ppmg.probability('d', list())))
    print('P(d|ra) = {} (should be 0.07142)'.format(ppmg.probability('d', list('ra'))))
    print('P(b|ar) = {}'.format(ppmg.probability('b', list('ar'))))
    print('P(r|ab) = {}'.format(ppmg.probability('r', list('ab'))))
    print('P(r|abcdrrrabc) = {}'.format(ppmg.probability('d', list('abcdrrrabc'))))
    print('P(d|rabcdddab) = {}'.format(ppmg.probability('d', list('rabcdddab'))))
    print('P(d|dd) = {}'.format(ppmg.probability('d', list('dd'))))

## References

- Ron Begleiter, Ran El-Yaniv, and Golan Yona. "On Prediction Using Variable
    Order Markov Models". In: Journal of Artificial Intelligence Research 22
    (2004), pp. 385-421.
- Moffat, A. (1990). Implementing the PPM data compression scheme. IEEE
    Transactions on Communications, 38(11), 1917–1921.
    partial string matching. IEEE Transactions on Communications, COM-32(4),
- Cleary, J., & Witten, I. (1984). Data compression using adaptive coding and
  396–402.
  


## Authors

Wannes Meert  
Dept. Computer Science, KU Leuven, Belgium  
https://dtai.cs.kuleuven.be  

Collaborators:
- Jeroen Mordijck
- Jesse Davis
