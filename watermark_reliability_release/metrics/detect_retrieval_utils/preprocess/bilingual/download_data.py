import os
import urllib.request
import sys

def check_site_exist(url):
    request = urllib.request.Request(url)
    request.get_method = lambda: 'HEAD'
    try:
        urllib.request.urlopen(request)
        return True
    except urllib.request.HTTPError:
        return False

def download_for_lang(lang):
    first = False
    if lang < "en":
        first = True

    cmd = "rm -Rf en-{0}".format(lang)
    os.system(cmd)

    cmd = "mkdir en-{0}".format(lang)
    os.system(cmd)

    """
    #download os
    if first:
        exists = check_site_exist("http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/{0}-en.txt.zip".format(lang))
        cmd = "wget http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/{0}-en.txt.zip -O en-{0}.os.txt.zip".format(lang)
    else:
        exists = check_site_exist("http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/en-{0}.txt.zip".format(lang))
        cmd = "wget http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/en-{0}.txt.zip -O en-{0}.os.txt.zip".format(lang)

    if exists:
        os.system(cmd)
        cmd = "unzip -d en-{0}/os/ en-{0}.os.txt.zip".format(lang)
        os.system(cmd)

    if lang == "zh":
        cmd = "wget http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/en-zh_cn.txt.zip -O en-zh.os.txt.zip"
        os.system(cmd)
        cmd = "unzip -d en-{0}/os/ en-{0}.os.txt.zip".format(lang)
        os.system(cmd)

    #download tanzil
    if first:
        exists = check_site_exist("http://opus.nlpl.eu/download.php?f=Tanzil/v1/moses/{0}-en.txt.zip".format(lang))
        cmd = "wget http://opus.nlpl.eu/download.php?f=Tanzil/v1/moses/{0}-en.txt.zip -O en-{0}.tz.txt.zip".format(lang)
    else:
        exists = check_site_exist("http://opus.nlpl.eu/download.php?f=Tanzil/v1/moses/en-{0}.txt.zip".format(lang))
        cmd = "wget http://opus.nlpl.eu/download.php?f=Tanzil/v1/moses/en-{0}.txt.zip -O en-{0}.tz.txt.zip".format(lang)

    if exists:
        os.system(cmd)
        cmd = "unzip -d en-{0}/tz/ en-{0}.tz.txt.zip".format(lang)
        os.system(cmd)

    #download europarl
    if first:
        exists = check_site_exist("http://opus.nlpl.eu/download.php?f=Europarl/v8/moses/{0}-en.txt.zip".format(lang))
        cmd = "wget http://opus.nlpl.eu/download.php?f=Europarl/v8/moses/{0}-en.txt.zip -O en-{0}.ep.txt.zip".format(lang)
    else:
        exists = check_site_exist("http://opus.nlpl.eu/download.php?f=Europarl/v8/moses/en-{0}.txt.zip".format(lang))
        cmd = "wget http://opus.nlpl.eu/download.php?f=Europarl/v8/moses/en-{0}.txt.zip -O en-{0}.ep.txt.zip".format(lang)

    if exists:
        os.system(cmd)
        cmd = "unzip -d en-{0}/ep/ en-{0}.ep.txt.zip".format(lang)
        os.system(cmd)

    #download un
    if first:
        exists = check_site_exist("http://opus.nlpl.eu/download.php?f=MultiUN/v1/moses/{0}-en.txt.zip".format(lang))
        cmd = "wget http://opus.nlpl.eu/download.php?f=MultiUN/v1/moses/{0}-en.txt.zip -O en-{0}.un.txt.zip".format(lang)
    else:
        exists = check_site_exist("http://opus.nlpl.eu/download.php?f=MultiUN/v1/moses/en-{0}.txt.zip".format(lang))
        cmd = "wget http://opus.nlpl.eu/download.php?f=MultiUN/v1/moses/en-{0}.txt.zip -O en-{0}.un.txt.zip".format(lang)

    if exists:
        os.system(cmd)
        cmd = "unzip -d en-{0}/un/ en-{0}.un.txt.zip".format(lang)
        os.system(cmd)
    """
    #download global voices
    if first:
        exists = check_site_exist("http://opus.nlpl.eu/download.php?f=GlobalVoices/v2017q3/moses/{0}-en.txt.zip".format(lang))
        cmd = "wget http://opus.nlpl.eu/download.php?f=GlobalVoices/v2017q3/moses/{0}-en.txt.zip -O en-{0}.gv.txt.zip".format(lang)
    else:
        exists = check_site_exist("http://opus.nlpl.eu/download.php?f=GlobalVoices/v2017q3/moses/en-{0}.txt.zip".format(lang))
        cmd = "wget http://opus.nlpl.eu/download.php?f=GlobalVoices/v2017q3/moses/en-{0}.txt.zip -O en-{0}.gv.txt.zip".format(lang)

    if exists:
        os.system(cmd)
        cmd = "unzip -d en-{0}/gv/ en-{0}.gv.txt.zip".format(lang)
        os.system(cmd)

if len(sys.argv) == 1:
    langs = ["fr", "de", "ru", "zh", "ar", "es", "tr"]
else:
    langs = sys.argv[1].split("-")

for i in langs:
    download_for_lang(i)

os.system("rm *.zip")