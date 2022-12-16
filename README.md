# BERTforQA
## Environment
### Google Colab
`CS1460_Final.ipynb` should run fine in google colab.

### Local
I'm using a `python3.8` virtual environment. To run different versions of
python alongside each other in Ubuntu, I used the [deadsnakes
PPA](https://linuxize.com/post/how-to-install-python-3-8-on-ubuntu-18-04/).
Once installed, use the following steps:\
`$ python3.8 -m venv [name of environment]`\
`$ source [name of environment]/bin/activate`\
`$ pip install torch datasets transformers`\
Now you should be able to run `CS1460_Final.py` locally.

## Running
### Google Colab
To run in google colab, simply connect to a GPU runtime and run all cells in
order.

### Local
To run locally, first activate your virtual environment. Then, you can just run\
`$ python CS11460_Final.py`

## Description
This code attempts to replicate the results of [Alberti et
al.](https://arxiv.org/pdf/1901.08634.pdf).  By using the
`DistilBertForQuestionAnswering` model configuration and initializing the
weights from `"distilbert-base-uncased"`, I fine-tuned the BERT model on a
subset of the Natural Questions dataset.  After three epochs, I obtained the
following results:\
\
(Cross-Entropy Loss) \
TRAINING LOSS: 0.884 \
VALIDATION LOSS: 1.556 \
\
PRECISION: 0.689 \
RECALL: 0.706 \
F1-SCORE: 0.654 \
\
