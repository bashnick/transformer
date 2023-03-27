# Transformer implementation from scratch
A codebase implementing a simple GPT-like model from scratch based on the [Attention is All You Need paper](https://arxiv.org/abs/1706.03762). The detailed theory explanation and a step by step guide on the Transformer architecture could be found in the article [Building a GPT-like Model from Scratch with Detailed Theory and Code Implementation](https://habr.com/en/company/ods/blog/708672/).

## Getting Started 
Follow [setup instructions here](requirements.txt) to get started.
```
$ git clone https://github.com/bashnick/transformer.git
$ cd transformer
$ conda create --name transformer python=3.9 -y
$ conda activate transformer
$ pip install -r requirements.txt
```

Alternatively you can install it using pyenv.
```
$ git clone https://github.com/bashnick/transformer.git
$ cd transformer
$ pyenv install 3.11.1
$ pyenv virtualenv 3.11.1 transformer
$ pyenv local transformer
$ pip install -r requirements.txt
```

## Data
Data is taken from the [DCEP: Digital Corpus of the European Parliament](https://joint-research-centre.ec.europa.eu/language-technology-resources/dcep-digital-corpus-european-parliament_en#Format%20and%20Structure%20of%20the%20Data). It has a variety of document types, from press releases to session and legislative documents related to European Parliament's activities and bodies. The current version of the corpus contains documents
that were produced between 2001 and 2012.

## Contributing
You are welcome to contribute to the repository with your PRs!

## License

The MIT License (MIT)
