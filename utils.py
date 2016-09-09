



class fileIterator(object):
    """ Class to iterate over a sharded Corpus.
    """
    def __init__(self, f, shards):
        self.corpus = []
        for i in range(0, shards):
            f_i = '%s-000%02d-of-00100' % (f, i)
            self.corpus.append(f_i)

    def __iter__(self):
        while self.corpus:
            corpus = open(self.corpus.pop(), 'r')
            for line in corpus:
                yield line.split()


