from typing import Dict, List, Set, Tuple, Optional, Any, Callable, NoReturn, Union, Mapping, Sequence, Iterable
from dataset_loader import load_wce_dataset, document_preprocess
from gensim.models import FastText

import multiprocessing
import pdb


def main():
    texts = load_wce_dataset()
    texts = [document_preprocess(text) for text in texts]
    model = FastText(vector_size=256, window=5, min_count=1, sentences=texts, epochs=10,
                     workers=multiprocessing.cpu_count())  # fasttext language model is trained
    model.save("checkpoints/fasttext.model")  # its checkpoint is then saved


if __name__ == "__main__":
    main()
