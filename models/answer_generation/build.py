
from sklearn.model_selection import train_test_split

import utils.data as datautils
from models.answer_generation import DATA_DIR, DATA_TEST, DATA_DEV, DATA_TRAIN, write_file
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

    if debug: print("second filtering")
    sf_items = datautils.filter_item_with_context(items)
    if debug: print(len(sf_items))
    if debug: print(_rel2count(sf_items))


    print("third filtering")
    cleaned_contexts_items = []
    for i in sf_items:
        new_ctx = datautils.simplify_ctx(i.context)
        if new_ctx == "":
            continue
        i.context = new_ctx
        i.question += " ?"
        cleaned_contexts_items.append(i)

    # if debug:
    #     for k in cleaned_contexts_items: print(k.context)

    # if debug: print(len(items), len(sf_items), len(cleaned_contexts_items))
    if debug: print(len(sf_items))
    if debug: print(_rel2count(sf_items))

    train, test = train_test_split(cleaned_contexts_items, test_size=0.33)

    if debug: print("train len: ", len(train))
    if debug: print("test len: ", len(test))

    # remember that each entry is an Item object; hence we need to unpack
    for data, filename in zip([train, test], [DATA_TRAIN, DATA_TEST]):
        write_file(data, filename)


if __name__ == '__main__':
    main()