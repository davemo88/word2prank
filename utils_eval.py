""" Utilities to evaluate Word Embeddings
"""
import pickle
import numpy as np
import scipy.stats as stats
import numpy.linalg as lina

class Evaluator(object):
    """
    """


    def set_embed(self, f):
        self.embed = pickle.load(open(f, 'r'))

    def getCosineSimilarity(self, w1, w2):
        v = self.embed[w1]
        w = self.embed[w2]
        return self.getCosineSimilarity_vec(v, w)

    def a_minus_b_plus_c(self, a, b, c):
        a = self.embed[a]
        b = self.embed[b]
        c = self.embed[c]
        return self.closest_word_vec(a-b+c)

    def getCosineSimilarity_vec(self, v, w):
        return v.dot(w) / (lina.norm(v)*lina.norm(w))


    def closest_word_vec(self, vec):
        sim = -1
        closest = vec
        for word, vector in self.embed.iteritems():
            cos = self.getCosineSimilarity_vec(vector, vec)
            if cos == 1:
                continue
            if cos > sim:
                sim = cos
                closest = word
        return sim, closest


    def ten_closest(self, word):
        return self.get_ten_closest(self.embed[word])


    def get_ten_closest(self, vec):
        closest = {}
        for _ in range(0, 10):
            sim = -1
            closest_word = ''
            for word, vector in self.embed.iteritems():
                if np.array_equal(vector, vec):
                    continue
                if word in closest:
                    continue
                cos = self.getCosineSimilarity_vec(vector, vec)
                if cos > sim:
                    closest_word = word
                    sim = cos
            closest[closest_word] = sim
        return closest


    def ten_farthest(self, word):
        return self.get_ten_farthest(self.embed[word])


    def get_ten_farthest(self, vec):
        farthest = {}
        for _ in range(0, 10):
            sim = 1
            farthest_word = 0
            for word, embedding in self.embed.iteritems():
                if word in farthest:
                    continue
                cos = self.getCosineSimilarity_vec(embedding, vec)
                if cos <= sim:
                    farthest_word = word
                    sim = cos
            farthest[farthest_word] = sim
        return farthest


    def closest_word(self, word):
        return self.closest_word_vec(self.embed[word])


    def farthest_word(self, word):
        sim = 1
        farthest = word
        for token in self.embed.keyss():
            cos = self.getCosineSimilarity(token, word)
            if cos < sim:
                sim = cos
                farthest = token
        return sim, farthest


    def evaluate(self, wordSims):
        sims_gold = []
        sims_embed = []
        not_found = 0
        for tup in wordSims:
            if tup[0] in self.embed and tup[1] in self.embed:
                sims_gold.append(tup[2])
                sims_embed.append(self.getCosineSimilarity(tup[0], tup[1]))
            else:
                print "Not Found: " + tup[0] + ' , ' + tup[1]
                not_found += 1
        print str(not_found) + " Word pairs were not found"
        return stats.spearmanr(sims_gold, sims_embed)

    def test(self):
        simsfile = open('./wordsim/combined.csv', 'r')
        line = simsfile.readline()
        print "First Line: " + line
        wordsimpairs = []
        for line in simsfile:
            words = line.split(',')
            wordsimpairs.append([words[0].lower(),
                                 words[1].lower(),
                                 float(words[2])])
        corr = self.evaluate(wordsimpairs)
        print "Evaluation on wordsim353 dataset: Evaluating on " + str(len(wordsimpairs)) + " Wordpairs."
        print "Spearman Rank Correlation: " + str(corr)


if __name__ == '__main__':
    EVAL = Evaluator()
    EVAL.set_embed('embed_matrix.pickle')
    EVAL.test()
    EVAL.set_embed('embed_matrix_idf.pickle')
    EVAL.test()
    print EVAL.closest_word('Obama')
    print EVAL.closest_word('mother')
    print EVAL.a_minus_b_plus_c('king', 'man', 'woman')
