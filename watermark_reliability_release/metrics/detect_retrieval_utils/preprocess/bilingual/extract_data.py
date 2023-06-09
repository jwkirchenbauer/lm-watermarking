import os
import sys
from os import path

def extract(lang):

    dir = "en-{0}".format(lang)

    key = "en-{0}".format(lang)
    if lang < "en":
        key = "{0}-en".format(lang)

    f_en = "en-{0}/ep/Europarl.{1}.en".format(lang, key)
    f_fr = "en-{0}/ep/Europarl.{1}.{0}".format(lang, key)
    os.system("echo -n \"\" > {0}/en.txt".format(dir))
    os.system("echo -n \"\" > {0}/fr.txt".format(dir))
    if path.exists(f_en):
        os.system("paste {0} {1} > {2}/paste.ep.txt".format(f_en, f_fr, dir))
        os.system("sort {0}/paste.ep.txt | uniq -u > {0}/sort.ep.txt".format(dir))
        os.system("shuf {0}/sort.ep.txt > {0}/shuf.ep.txt".format(dir))
        os.system("python ../filter_length.py {0}/shuf.ep.txt {0}/shuf.clean.ep.txt".format(dir))
        os.system("/bin/bash -c \"cut -f 1 {0}/shuf.clean.ep.txt >> {0}/en.txt\"".format(dir))
        os.system("/bin/bash -c \"cut -f 2 {0}/shuf.clean.ep.txt >> {0}/fr.txt\"".format(dir))

    f_en = "en-{0}/os/OpenSubtitles.{1}.en".format(lang, key)
    f_fr = "en-{0}/os/OpenSubtitles.{1}.{0}".format(lang, key)
    if path.exists(f_en):
        os.system("paste {0} {1} > {2}/paste.os.txt".format(f_en, f_fr, dir))
        os.system("sort {0}/paste.os.txt | uniq -u > {0}/sort.os.txt".format(dir))
        os.system("shuf {0}/sort.os.txt > {0}/shuf.os.txt".format(dir))
        os.system("python ../filter_length.py {0}/shuf.os.txt {0}/shuf.clean.os.txt".format(dir))
        os.system("/bin/bash -c \"cut -f 1 {0}/shuf.clean.os.txt | head -n 4000000 >> {0}/en.txt\"".format(dir))
        os.system("/bin/bash -c \"cut -f 2 {0}/shuf.clean.os.txt | head -n 4000000 >> {0}/fr.txt\"".format(dir))

    f_en = "en-{0}/tz/Tanzil.{1}.en".format(lang, key)
    f_fr = "en-{0}/tz/Tanzil.{1}.{0}".format(lang, key)
    if path.exists(f_en):
        os.system("paste {0} {1} > {2}/paste.tz.txt".format(f_en, f_fr, dir))
        os.system("sort {0}/paste.tz.txt | uniq -u > {0}/sort.tz.txt".format(dir))
        os.system("shuf {0}/sort.tz.txt > {0}/shuf.tz.txt".format(dir))
        os.system("python ../filter_length.py {0}/shuf.tz.txt {0}/shuf.clean.tz.txt".format(dir))
        os.system("/bin/bash -c \"cut -f 1 {0}/shuf.clean.tz.txt >> {0}/en.txt\"".format(dir))
        os.system("/bin/bash -c \"cut -f 2 {0}/shuf.clean.tz.txt >> {0}/fr.txt\"".format(dir))

    f_en = "en-{0}/gv/GlobalVoices.{1}.en".format(lang, key)
    f_fr = "en-{0}/gv/GlobalVoices.{1}.{0}".format(lang, key)
    if path.exists(f_en):
        os.system("paste {0} {1} > {2}/paste.gv.txt".format(f_en, f_fr, dir))
        os.system("sort {0}/paste.gv.txt | uniq -u > {0}/sort.gv.txt".format(dir))
        os.system("shuf {0}/sort.gv.txt > {0}/shuf.gv.txt".format(dir))
        os.system("python ../filter_length.py {0}/shuf.gv.txt {0}/shuf.clean.gv.txt".format(dir))
        os.system("/bin/bash -c \"cut -f 1 {0}/shuf.clean.gv.txt >> {0}/en.txt\"".format(dir))
        os.system("/bin/bash -c \"cut -f 2 {0}/shuf.clean.gv.txt >> {0}/fr.txt\"".format(dir))

    if lang == "ar" or lang == "ru" or lang == "zh":
        f_en = "en-{0}/un/MultiUN.{1}.en".format(lang, key)
        f_fr = "en-{0}/un/MultiUN.{1}.{0}".format(lang, key)
        if path.exists(f_en):
            os.system("paste {0} {1} > {2}/paste.un.txt".format(f_en, f_fr, dir))
            os.system("sort {0}/paste.un.txt | uniq -u > {0}/sort.un.txt".format(dir))
            os.system("shuf {0}/sort.un.txt > {0}/shuf.un.txt".format(dir))
            os.system("python ../filter_length.py {0}/shuf.un.txt {0}/shuf.clean.un.txt".format(dir))
            os.system("/bin/bash -c \"cut -f 1 {0}/shuf.clean.un.txt | head -n 4000000 >> {0}/en.txt\"".format(dir))
            os.system("/bin/bash -c \"cut -f 2 {0}/shuf.clean.un.txt | head -n 4000000 >> {0}/fr.txt\"".format(dir))
            if lang == "ar":
                os.system("/bin/bash -c \"cut -f 1 {0}/shuf.clean.un.txt | head -n 4000000 >> "
                          "multi.zero.all.txt\"".format(dir))

    if lang == "zh":
        os.system("python ../jieba_tokenize.py {0}/fr.txt".format(dir))

    if lang == "es":
        os.system("cat {0}/en.txt >> multi.zero.all.txt".format(dir))

    os.system("cat {0}/fr.txt >> multi.zero.all.txt".format(dir))

os.system("echo -n \"\" > multi.zero.all.txt")

if len(sys.argv) == 1:
    langs = ["fr", "de", "ru", "zh", "ar", "es", "tr"]
else:
    langs = sys.argv[1].split("-")

for i in langs:
    extract(i)
