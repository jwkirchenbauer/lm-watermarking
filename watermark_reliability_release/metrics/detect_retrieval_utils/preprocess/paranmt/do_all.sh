
# bash doèall.sh 0.7 1.0 0.5

simlow=$1
simhigh=$2
ovl=$3

rm paranmt-sim-low=$1-sim-high=$2-ovl=$3.txt
rm -Rf scratch
mkdir scratch

if [ ! -d "../mosesdecoder" ]
then
    git clone https://github.com/moses-smt/mosesdecoder.git
    mv mosesdecoder ..
fi

if [ ! -f "para-nmt-50m.txt" ]
then
    wget http://phontron.com/data/para-nmt-50m.zip
    unzip para-nmt-50m.zip
    rm para-nmt-50m.zip
fi

bash add_language_labels.sh
python -u add_overlap_labels.py

python extract_data.py --cutoff-sim-low $1 --cutoff-sim-high $2 --cutoff-ovl $3

python preprocess_data.py --lower-case 1 --paranmt-file scratch/paranmt.sim-low=$1-sim-high=$2-ovl=$3.txt --name "sim-low=$1-sim-high=$2-ovl=$3"

python ../text2HDF5.py scratch/paranmt.sim-low=$1-sim-high=$2-ovl=$3.final.txt 4
mv scratch/paranmt.sim-low=$1-sim-high=$2-ovl=$3.final.txt .
mv scratch/paranmt.sim-low=$1-sim-high=$2-ovl=$3.final.h5 .
mv scratch/paranmt.sim-low=$1-sim-high=$2-ovl=$3.final.vocab .