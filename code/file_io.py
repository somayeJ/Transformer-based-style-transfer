
def load_sent(path, max_size=-1):
    data = []
    with open(path) as f:
        for line in f:
            if len(data) == max_size:
                break
            data.append(line.split())
    return data, len(data)

def load_vec(path):
    x = []
    with open(path) as f:
        for line in f:
            p = line.split()
            p = [float(v) for v in p]
            x.append(p)
    return x

def write_doc(docs, sents, path):
    with open(path, 'w') as f:
        index = 0
        for doc in docs:
            for i in range(len(doc)):
                f.write(' '.join(sents[index]))
                f.write('\n' if i == len(doc)-1 else ' ')
                index += 1


def write_vec(vecs, path):
    with open(path, 'w') as f:
        for vec in vecs:
            for i, x in enumerate(vec):
                f.write('%.3f' % x)
                f.write('\n' if i == len(vec)-1 else ' ')

def write_sent(sents, path):
    with open(path, 'w') as f:
        for sent in sents:
            #f.write(' '.join(sent) + '\n')
            f.write(sent + '\n')

