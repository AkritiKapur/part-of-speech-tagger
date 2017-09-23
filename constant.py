import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
DOC_PATH = SRC + '/data/'
INPUT_PATH = SRC + '/input/'
OUTPUT_PATH = SRC + '/output/'

TRAINING_DATA_FILE = 'berp-POS-training.txt'
TEST_DATA_FILE = 'berp-POS-test.txt'
WORD_TAG_FILE = 'word-tag-count.txt'
UNIGRAM_TAG_FILE = 'unigram-counts.txt'
BIGRAM_TAG_FILE = 'bigram-tag-counts.txt'
WORD_FILE = 'word.txt'
