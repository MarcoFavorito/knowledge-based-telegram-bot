from models.ModelManager import ModelManager
from models.relation_classifier import DATA_DIR, split_line, MODEL_PATH, read_file, DATA_DEV, DATA_TRAIN
from models.relation_classifier.RelationClassifierRNNBased import RelationClassifierRNNBased


def main():
    X_train, Y_train = read_file(DATA_TRAIN)
    X_dev, Y_dev = read_file(DATA_DEV)

    rc = RelationClassifierRNNBased()
    rc.make_vocab(X_train)
    rc.train(X_train, Y_train, dev=(X_dev, Y_dev))

    rc.save(MODEL_PATH)



if __name__ == '__main__':
    main()