# Text Segmentation as a Supervised Learning Task

This repository contains code and supplementary materials which are required to train and evaluate a model as described in the paper [Text Segmentation as a Supervised Learning Task](https://arxiv.org/abs/1803.09337)

## Fork Notice

Firstly, I want to acknowledge that this fork heavily builds upon the code from the original repository, with the changes mentioned in [18](https://github.com/koomri/text-segmentation/pull/18). 
I have implemented several performance optimizations and code improvements to enhance usability, as outlined in the commits.

Due to resource constraints, I could only train the model for a **single epoch on Google Colab**. If anyone is able to **replicate the full results during testing**, your contributions and feedback would be greatly appreciated.

I plan to spend time adding features and further optimizing the code as I find the opportunity. Below is a **To-Do List** of the upcoming enhancements I aim to implement:

### To-Do List:
- [ ] Add support for **Multi-GPU** training.
- [ ] Add **TensorBoard logging** for better tracking and visualization.

If you have any suggestions or ideas for additional features or improvements, feel free to raise an issue or submit a pull request!


## Download required resources

wiki-727K, wiki-50 datasets:
>  https://www.dropbox.com/sh/k3jh0fjbyr0gw0a/AADzAd9SDTrBnvs1qLCJY5cza?dl=0

word2vec:
>  https://drive.google.com/a/audioburst.com/uc?export=download&confirm=zrin&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM



Fill relevant paths in configgenerator.py, and execute the script (git repository includes Choi dataset)

## Creating an environment:

    conda create -n textseg python=3.10
    source activate textseg
    pip install -r requirements.txt
    pip install tqdm pathlib2 segeval tensorboard_logger flask flask_wtf nltk
    pip install pandas xlrd xlsxwriter termcolor

## How to run training process?

    python run.py --help

Example:

    python run.py --cuda --model max_sentence_embedding --wiki 

## How to evaluate trained model (on wiki-727/choi dataset)?

    python test_accuracy.py  --help

Example:

    python test_accuracy.py --cuda --model <path_to_model> --wiki



## How to create a new wikipedia dataset:
    python wiki_processor.py --input <input> --temp <temp_files_folder> --output <output_folder> --train <ratio> --test <ratio>

Input is the full path to the wikipedia dump, temp is the path to the temporary files folder, and output is the path to the newly generated wikipedia dataset.

Wikipedia dump can be downloaded from following url:

> https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
