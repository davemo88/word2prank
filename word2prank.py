""" ??
"""
import numpy as np
import pickle
import click
import time
from collections import defaultdict
import numpy.linalg as lina
import sys
import utils



@click.group()
def word2prank():
    pass


@word2prank.command()
@click.option('--threshold', default=100, help='Min Frequency of words')
@click.option("--corpusfile", help='loc of corpusfile')
@click.option('--outfile', default='adj_matrix.pickle',
              help='Counts Output File')
@click.option('--vocfile', default='vocab.pickle', help='Vocab Output File')
@click.option('--c_length', default=3, help='Length of Contexts')
@click.option('--num_shards', default=1, help='Number of shards to use')
def makematrix(**kwargs):
    """
    ?
    """
    corpusfile = kwargs['corpusfile']
    outfile = kwargs['outfile']
    c_length = kwargs['c_length']
    threshold = kwargs['threshold']
    num_shards = kwargs['num_shards']
    vocfile = kwargs['vocfile']
    # Generate Counts of each distinct word
    print 'Generating Wordcounts...'
    it = utils.fileIterator(corpusfile, num_shards, 100000)
    counts = defaultdict(int)
    num_sentences = 0
    vocab = set()
    for sentence in it:
        num_sentences += 1
        for word in sentence:
            counts[word] += 1.0
    print 'Counts generated from %d sentences.' % num_sentences
    print 'Initializing Adj Matrix...'
    # Generate Vocabulary Mapping. Disregard uncommon words.
    vocab = {word: idx for idx, word in
             enumerate([word for word, count in counts.iteritems()
                        if count > threshold])}
    vocab_size = len(vocab.keys())
    # Initialize Adj matrix
    adj = [defaultdict(int) for _ in range(vocab_size)]
    print 'Matrix initialized with vocab size of {0}.'.format(vocab_size)
    # Generate Contexts for every Sentence in Corpus
    it = utils.fileIterator(corpusfile, num_shards, 100000)
    print 'Starting Matrix Computation...'
    sentence_count = 0
    for sentence in it:
        sentence_count += 1
        if ((sentence_count % (num_sentences / 100)) == 0):
            sys.stderr.write(
                'Progress: {0}%\r'.format((100*sentence_count) / num_sentences))
        # Generate Contexts for each word in sentence
        for pos in range(0, len(sentence)):
            word = sentence[pos]
            # Disregard uncommon Words
            if word not in vocab:
                continue
            word = vocab[word]
            # For each context 
            for c in (-c_length,c_length):
                # Out Of Range
                if c == 0 or  pos+c < 0 or pos+c >= len(sentence):
                    continue
                context = sentence[pos+c]
                # Disregard Uncommon Contexts
                if context not in vocab:
                    continue
                context = vocab[context]
                adj[word][context] += 1.0
    # Normalize Context Counts
    for word in range(0,vocab_size):
        total_counts = sum(adj[word].values())
        adj[word] = {context : count / total_counts
                     for context, count in adj[word].iteritems()}
    pickle.dump(adj, open(outfile, 'w'))
    pickle.dump(vocab, open(vocfile, 'w'))
    return


@word2prank.command()
@click.option("--countsfile", help='loc of countsfile')
@click.option('--outfile', default='embeddings.pickle', help='Output File')
@click.option('--iterations', default=6, help='Number of Iterations')
@click.option('--dimension', default=100, help='Embedding Dimension')
def w2p(countsfile, outfile, dimension, iterations):
    """

    """
    counts = pickle.load(open(countsfile,'r'))
    vocab_size = len(counts)
    ## Initialize Embedding Matrix Uniformly from [-0.5/dim,0.5/dim]^dim
    print 'Initializing Embeddings for {0} words...'.format(num_words)
    embed = (np.random.rand(vocab_size, dimension) - 0.5) / dimension
    for it in range(iterations):
        print 'Starting iteration {0}...'.format(it)
        words_done = 0
        next_embed = np.zeros(vocab_size, dimension)
        for word in range(vocab_size):
            words_done += 1
            if ((words_done % (vocab_size / 100)) == 0):
                sys.stderr.write(
                    'Progress: {0}%\r'.format((100*words_done) / num_words))
            contexts = counts[word]
            for context, weight in contexts.iteritems():
                next_embed[word] += embed[context] * weight 
        print 'Finished Iteration {0}'.format(it)
        # Print some Stats
        avg_dist = 0
        max_dist = 0
        for word in rang(vocab_size):
            dist = lina.norm(embed[word]-next_embed[word])
            avg_dist += dist
            if dist > max_dist:
                max_dist = dist
        avg_dist /= num_words
        print 'Avg. L2 Dist: {0}, Max L2 Dist: {1}'.format(avg_dist, max_dist)
        embed = next_embed
    pickle.dump(embed, open(outfile,'w'))

        
if __name__ == "__main__":
    word2prank()
