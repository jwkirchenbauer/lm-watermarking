import sentencepiece as spm
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--paranmt-file')
parser.add_argument('--name')
parser.add_argument('--lower-case', type=int, default=1)

args = parser.parse_args()

def encode_sp(f, fout, sp_model):
    f = open(f, 'r')
    lines = f.readlines()
    fout = open(fout, 'w')

    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model)
    for line in lines:
        if args.lower_case:
            line = line.lower()
        arr = line.strip().split('\t')
        s0 = sp.EncodeAsPieces(arr[0].strip())
        s0 = " ".join(s0)
        s1 = sp.EncodeAsPieces(arr[1].strip())
        s1 = " ".join(s1)
        arr[0] = s0
        arr[1] = s1
        fout.write("\t".join(arr) + "\n")
    
    f.close()
    fout.close()

os.system("cut -f 1 {0} > {1}.all.txt".format(args.paranmt_file, args.paranmt_file.replace(".txt","")))
os.system("cut -f 2 {0} >> {1}.all.txt".format(args.paranmt_file, args.paranmt_file.replace(".txt","")))

if args.lower_case:
    os.system(
        "perl ../mosesdecoder/scripts/tokenizer/lowercase.perl < {0}.all.txt > {0}.temp".format(args.paranmt_file.replace(".txt", "")))
    os.system("mv {0}.temp {0}.all.txt".format(args.paranmt_file.replace(".txt", "")))

spm.SentencePieceTrainer.Train('--input={0}.all.txt --model_prefix=paranmt.{1} --vocab_size=50000 '
                               '--character_coverage=0.995 --input_sentence_size=10000000 --hard_vocab_limit=false'.format(args.paranmt_file.replace(".txt", ""), args.name))
encode_sp(args.paranmt_file, "{0}.final.txt".format(args.paranmt_file.replace(".txt", "")), 'paranmt.{0}.model'.format(args.name))
