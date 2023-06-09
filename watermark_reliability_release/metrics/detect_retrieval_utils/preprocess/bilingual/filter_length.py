import sys

f = open(sys.argv[1], 'r')
lines = f.readlines()
f.close()

output = []

for line in lines:
    i = line.strip().lower()
    i = i.split('\t')
    en = i[0]
    l = en.split()
    if len(l) < 3 or len(l) > 100:
        continue
    else:
        output.append(line.lower())

fout = open(sys.argv[2], 'w')
for i in output:
    fout.write(i)
fout.close()
