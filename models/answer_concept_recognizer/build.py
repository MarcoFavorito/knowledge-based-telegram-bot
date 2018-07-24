
from sklearn.model_selection import train_test_split

import constants as c
import utils.data as datautils
import numpy as np
from keras.models import load_model
from utils.dependency_parser import get_spacy_parser
from models.answer_concept_recognizer import DATA_DIR, DATA_TEST, DATA_DEV, DATA_TRAIN, write_file
import utils.graph_utils as gut

def _rel2count(items):
    rel2count = {}
    for i in items:
        if not i.relation in rel2count: rel2count[i.relation] = 0
        rel2count[i.relation] += 1
    return rel2count

def main():
    debug = True

    if debug: print("from db (with filtering)")
    items = datautils.get_items_from_db(max_per_relation=2000, filtering=True)
    if debug: print(len(items))
    if debug: print(_rel2count(items))

    sf_items = list(filter(datautils.filter_item_by_answer, items))
    train, test = train_test_split(sf_items, test_size=0.33)
    train, dev = train_test_split(train, test_size=0.33)


    if debug: print("train len: ", len(train))
    if debug: print("dev len: ", len(dev))
    if debug: print("test len: ", len(test))

    # remember that each entry is an Item object; hence we need to unpack
    for data, filename in zip([train, dev, test], [DATA_TRAIN, DATA_DEV, DATA_TEST]):
        write_file(data, filename)


if __name__ == '__main__':
    main()