
# bash doèall.sh fr-es

rm -Rf "en-"$1
mkdir "en-"$1

if [ ! -d "../mosesdecoder" ]
then
    git clone https://github.com/moses-smt/mosesdecoder.git
    mv mosesdecoder ..
fi

cd "en-"$1 || exit

python -u ../download_data.py $1
python -u ../extract_data.py $1

langs=$(echo $1 | tr "-" "\n")

for l in $langs
do
    python -u ../preprocess_data.py --lang $l --dir en-$l
    python -u ../../text2HDF5.py train-$l-en.txt 2
done
