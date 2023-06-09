import argparse
import random

random.seed(1)

parser = argparse.ArgumentParser()

parser.add_argument("--cutoff-sim-low", default=0.7, type=float, help='cutoff for sim')
parser.add_argument("--cutoff-sim-high", default=1.0, type=float, help='cutoff for sim')
parser.add_argument("--cutoff-ovl", default=0.5, type=float, help='cutoff for overlap')

args = parser.parse_args()

f = open("scratch/para-nmt-50m-labeled-overlap.txt", "r")
lines = f.readlines()
random.shuffle(lines)

ct = 0
fout = open('scratch/paranmt.sim-low={0}-sim-high={1}-ovl={2}.txt'.format(args.cutoff_sim_low, args.cutoff_sim_high,
                                                                          args.cutoff_ovl), 'w')

for line in lines:
    i = line.strip()
    sim = i.split('\t')[2]
    sim = float(sim)
    ovl = i.split('\t')[3]
    ovl = float(ovl)
    s1 = i.split('\t')[0]
    s2 = i.split('\t')[1]
    toks = len(s1.split()) + len(s2.split())
    toks = toks / 2
    if toks < 5 or toks > 40:
        continue
    label1 = i.split('\t')[4]
    label2 = i.split('\t')[5]
    if label1.split()[0] == "__label__en" and label2.split()[0] == "__label__en":
        if sim >= args.cutoff_sim_low and sim <= args.cutoff_sim_high and ovl <= args.cutoff_ovl:
            arr = line.split('\t')[0:4]
            fout.write("\t".join(arr).strip()+"\n")
            ct += 1
fout.close()
