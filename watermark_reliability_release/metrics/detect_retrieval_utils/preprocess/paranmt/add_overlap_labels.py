import nltk

def get_overlap(t, r, type):
    if type == 2:
        temp = []
        for i in range(len(t) - 1):
            temp.append(t[i] + " " + t[i + 1])
        t = temp
        temp = []
        for i in range(len(r) - 1):
            temp.append(r[i] + " " + r[i + 1])
        r = temp
    elif type == 3:
        temp = []
        for i in range(len(t) - 2):
            temp.append(t[i] + " " + t[i + 1] + " " + t[i + 2])
        t = temp
        temp = []
        for i in range(len(r) - 2):
            temp.append(r[i] + " " + r[i + 1] + " " + r[i + 2])
        r = temp
    if len(r) < len(t):
        start = r
        end = t
    else:
        start = t
        end = r
    start = list(start)
    den = len(start)
    if den == 0:
        return 0.
    num = 0
    for i in range(len(start)):
        if start[i] in end:
            num += 1
    return float(num) / den

fout = open('scratch/para-nmt-50m-labeled-overlap.txt', 'w')

with open('scratch/para-nmt-50m-labeled.txt', 'r') as f:
    for i in f:
        arr = i.strip().split('\t')
        g = arr[0]
        tr = arr[1]
        sim = arr[2]
        l1 = arr[3]
        l2 = arr[4]
        gtoks = nltk.word_tokenize(g.strip().lower())
        gtr = nltk.word_tokenize(tr.strip().lower())
        o = get_overlap(gtoks, gtr, 3)
        st = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(g, tr,  sim, o, l1, l2)
        fout.write(st)
fout.close()
