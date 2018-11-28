# IARM
This repo contains the source code of the paper --

[IARM: Inter-Aspect Relation Modeling with Memory Networks in Aspect-Based Sentiment Analysis](http://aclweb.org/anthology/D18-1377).
Navonil Majumder, Soujanya Poria, Alexander Gelbukh, Md. Shad Akhtar, Erik Cambria, Asif Ekbal. EMNLP 2018

This method attempts to model the relationship among the different aspect-terms in a sentence using _memory networks_ to enable better sentiment classification of the aspects.

## Requirements

- Python 2.7
- PyTorch 0.3
- Keras 1.0

## Execution

Execute the file `ABSA-emb-gpu-final-newarch3.py` for training and testing on SemEval 2014 ABSA dataset.
The following are the command-line arguments:
- `--no-cuda`: GPU is not used
- `--lr`: set learning rate
- `--l2`: set L2-norm weight
- `--batch-size`: set batch size
- `--epochs`: set number of epochs
- `--hops`: set number hops of memory network
- `--hidden-size`: set hidden representation size
- `--output-size`: set output representation size
- `--dropout-p`: set dropout probability
- `--dropout-lstm`: set recurrent dropout probability
- `--dataset`: set which dataset to use - `Restaurants` or `Laptop`

Example:
```
python ABSA-emb-gpu-final-newarch3.py --lr 0.0001 --l2 0.0001 --dataset Laptop --hops 3 --epochs 30 --hiddem-size 400 --output-size 300 --dropout-p 0.1 --dropout-lstm 0.2
```
## Citation
If you find this code useful please cite the following in your work:
```
@InProceedings{D18-1377,
  author = 	"Majumder, Navonil
		and Poria, Soujanya
		and Gelbukh, Alexander
		and Akhtar, Md Shad
		and Cambria, Erik
		and Ekbal, Asif",
  title = 	"IARM: Inter-Aspect Relation Modeling with Memory Networks in Aspect-Based Sentiment Analysis",
  booktitle = 	"Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"3402--3411",
  location = 	"Brussels, Belgium",
  url = 	"http://aclweb.org/anthology/D18-1377"
}
```
