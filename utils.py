



class fileIterator(object):
    def __init__(self, f, shards, buf):
        self.buf = buf
        self.f = []
        for i in range(0, shards):
            f_i = '%s-000%02d-of-00100' % (f,i)
            self.f.append(f_i)

    def __iter__(self):	      
        f = open(self.f.pop(), 'r')
        data = []
        num_sen = 0
        while True:
            line = f.readline()
            if not line:
                if not self.f:
                    break
                f = open(self.f.pop(), 'r')
                continue
            num_sen += 1
            data.append(line.split())
            if num_sen >= self.buf:
                for sentence in data:
                    yield sentence
                num_sen = 0
                data = []


