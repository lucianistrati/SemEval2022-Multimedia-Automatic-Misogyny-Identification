from gensim.models import FastText
from dataset_loader import *
import multiprocessing
texts = load_wce_dataset()
texts = [document_preprocess(text) for text in texts]
model = FastText(vector_size=256, window=5, min_count=1, sentences=texts, epochs=10, workers=multiprocessing.cpu_count()) # fasttext language model is trained
model.save("fasttext.model") # its checkpoint is then saved
