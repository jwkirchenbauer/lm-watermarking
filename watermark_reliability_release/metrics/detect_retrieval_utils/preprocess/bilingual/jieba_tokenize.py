import sys
import jieba

f = sys.argv[1]

f = open(f, 'r')
lines = f.readlines()

fout = open(sys.argv[1], "w")
for i in lines:
    newl = jieba.cut(i.strip(), cut_all=True)
    newl = " ".join(newl)
    fout.write(newl + "\n")
fout.close()    
