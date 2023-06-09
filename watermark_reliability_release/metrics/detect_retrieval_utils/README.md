# Paraphrastic Representations at Scale

Code to train models from "Paraphrastic Representations at Scale".

The code is written in Python 3.7 and requires H5py, jieba, numpy, scipy, sentencepiece, sacremoses, and PyTorch >= 1.0 libraries. These can be insalled with the following command:

    pip install -r requirements.txt

To get started, download the pre-trained models, data files used for training, and the STS evaluation data:

    wget http://phontron.com/data/paraphrase-at-scale.zip
    unzip paraphrase-at-scale.zip
    rm paraphrase-at-scale.zip
    wget http://www.cs.cmu.edu/~jwieting/STS.zip .
    unzip STS.zip
    rm STS.zip

To download only the English model trained on ParaNMT:

    wget http://www.cs.cmu.edu/~jwieting/paraphrase-at-scale-english.zip .

If you use our code, models, or data for your work please cite:

    @article{wieting2021paraphrastic,
        title={Paraphrastic Representations at Scale},
        author={Wieting, John and Gimpel, Kevin and Neubig, Graham and Berg-Kirkpatrick, Taylor},
        journal={arXiv preprint arXiv:2104.15114},
        year={2021}
    }

    @inproceedings{wieting19simple,
        title={Simple and Effective Paraphrastic Similarity from Parallel Translations},
        author={Wieting, John and Gimpel, Kevin and Neubig, Graham and Berg-Kirkpatrick, Taylor},
        booktitle={Proceedings of the Association for Computational Linguistics},
        url={https://arxiv.org/abs/1909.13872},
        year={2019}
    }

To embed a list of sentences:

    python -u embed_sentences.py --sentence-file paraphrase-at-scale/example-sentences.txt --load-file paraphrase-at-scale/model.para.lc.100.pt  --sp-model paraphrase-at-scale/paranmt.model --output-file sentence_embeds.np --gpu 0
    
To score a list of sentence pairs:

    python -u score_sentence_pairs.py --sentence-pair-file paraphrase-at-scale/example-sentences-pairs.txt --load-file paraphrase-at-scale/model.para.lc.100.pt  --sp-model paraphrase-at-scale/paranmt.model --gpu 0

To train a model (for example, on ParaNMT):

    python -u main.py --outfile model.para.out --lower-case 1 --tokenize 0 --data-file paraphrase-at-scale/paranmt.sim-low=0.4-sim-high=1.0-ovl=0.7.final.h5 \
           --model avg --dim 1024 --epochs 25 --dropout 0.0 --sp-model paraphrase-at-scale/paranmt.model --megabatch-size 100 --save-every-epoch 1 --gpu 0 --vocab-file paraphrase-at-scale/paranmt.sim-low=0.4-sim-high=1.0-ovl=0.7.final.vocab

To download and preprocess raw data for training models (both bilingual and ParaNMT), see preprocess/bilingual and preprocess/paranmt.
