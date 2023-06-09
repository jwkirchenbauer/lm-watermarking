import os
import sentencepiece as spm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default="en-ar",
                    help='')
parser.add_argument('--lang', default='fr',
                    help='')
parser.add_argument('--lower-case', type=int, default=1)
args = parser.parse_args()

fr_file = args.dir + "/fr.txt"
en_file = args.dir + "/en.txt"
lang = args.lang
lower_case = args.lower_case

f_fr = open(fr_file, 'r')
lines_fr = f_fr.readlines()

f_en = open(en_file, 'r')
lines_en = f_en.readlines()

def encode_sp(lines_fr, lines_en, fout, sp_model):
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model)
    output = []
    for i in zip(lines_fr, lines_en):
        i0 = i[0].strip()
        if args.lower_case:
            i0 = i0.lower()
        s0 = sp.EncodeAsPieces(i0)
        if len(s0) > 100:
            s0 = s0[0:100]
        s0 = " ".join(s0)

        i1 = i[1].strip()
        if args.lower_case:
            i1 = i1.lower()
        s1 = sp.EncodeAsPieces(i1)
        if len(s1) > 100:
            s1 = s1[0:100]
        s1 = " ".join(s1)
        output.append(s0 + "\t" + s1)

    fout = open(fout, "w")
    for i in output:
        fout.write(i + "\n")
    fout.close()

if args.lower_case:
    os.system("cat {0} {1} > {2}/all.{3}.txt".format(fr_file, en_file, args.dir, lang))
    os.system(
        "perl ../../mosesdecoder/scripts/tokenizer/lowercase.perl < {0}/all.{1}.txt > {0}/all.{1}.lc.txt".format(args.dir, lang))
    spm.SentencePieceTrainer.Train(
        '--input={0}/all.{1}.lc.txt --model_prefix={1}.lc.sp.50k --vocab_size=50000 --character_coverage=0.995 --hard_vocab_limit=false --input_sentence_size=10000000'.format(
            args.dir, lang))
    encode_sp(lines_fr, lines_en, "train-{0}-en.txt".format(lang),
              '{0}.lc.sp.50k.model'.format(lang))
else:
    os.system("cat {0} {1} > {2}/all.{3}.txt".format(fr_file, en_file, args.dir, lang))
    spm.SentencePieceTrainer.Train(
        '--input={0}/all.{1}.txt --model_prefix={1}.sp.50k --vocab_size=50000 --character_coverage=0.995 --hard_vocab_limit=false --input_sentence_size=10000000'.format(
            args.dir, lang))
    encode_sp(lines_fr, lines_en, "train-{0}-en.txt".format(lang), '{0}.sp.50k.model'.format(lang))
