from models.ModelManager import ModelManager
from models.concept_recognizer import DATA_DIR, split_line, MODEL_PATH, read_file, DATA_DEV, DATA_TRAIN
from models.concept_recognizer.ConceptRecognizerRNNBased import ConceptRecognizerRNNBased


def main():
    """
    X_train:
        list of (question, relation)
    Y_train:
        list of (c1::babelnet_id, c2::babelnet_id)

    """

    X_train, Y_train = read_file(DATA_TRAIN)
    X_dev, Y_dev = read_file(DATA_DEV)

    ag = ConceptRecognizerRNNBased()
    ag.make_vocab(map(lambda x: x[0], X_train))
    ag.train(X_train, Y_train, dev=(X_dev, Y_dev))

    ag.save(MODEL_PATH)



if __name__ == '__main__':
    main()