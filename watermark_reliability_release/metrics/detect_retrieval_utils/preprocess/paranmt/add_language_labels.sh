if [ ! -d "fastText-0.9.2" ]
then
    wget https://github.com/facebookresearch/fastText/archive/v0.9.2.zip
    unzip v0.9.2.zip
    cd fastText-0.9.2
    make
    wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin .
    cd ..
    rm v0.9.2.zip
fi

cd fastText-0.9.2
cut -f 1 ../para-nmt-50m.txt > ../scratch/para-nmt-50m-1st-col.txt
cut -f 2 ../para-nmt-50m.txt > ../scratch/para-nmt-50m-2nd-col.txt
./fasttext predict lid.176.bin ../scratch/para-nmt-50m-1st-col.txt 2 > ../scratch/col1.pred
./fasttext predict lid.176.bin ../scratch/para-nmt-50m-2nd-col.txt 2 > ../scratch/col2.pred
paste ../para-nmt-50m.txt ../scratch/col1.pred ../scratch/col2.pred | sort | uniq -u > ../scratch/para-nmt-50m-labeled.txt
