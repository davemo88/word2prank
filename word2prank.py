"""
    ?
"""

import numpy
import click
import pickle

@click.argument('countsfile')
@click.option('--outfile', default='results.pkl', help='output file')
@click.option('--dimension', default=100, help='dimension of embeddings')
@click.option('--iterations', default=10, help='number of iterations')
def word2prank(countsfile, outfile, dimension, iterations):
    """

    """

    counts = pickle.loads(countsfile)
    vocab = counts.keys()
    num_words = len(vocab)
    reverse_vocab = {vocab[i] : i for i in range(num_words)}
## randomly initialize our Embedding Matrix
    em = numpy.random.rand(num_words, dimension)

    for i in range(iterations):
        next_em = numpy.zeros((num_words, dimension))
        for j in range(num_words):
            for word, count in counts[vocab[j]]:
                next_em[j] += em[reverse_vocab[word]] * count

        em = next_em

    word_embeddings = {word : em[reverse_vocab[word]] for word in vocab}
    pickle.dumps(em, word_embeddings)
