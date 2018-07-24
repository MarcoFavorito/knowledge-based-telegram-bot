from models.ModelManager import ModelManager
from models.relation_classifier import read_file, DATA_DIR, DATA_TEST, MODEL_PATH
from models.relation_classifier.RelationClassifierRNNBased import RelationClassifierRNNBased


def main():
    X_test, Y_test = read_file(DATA_TEST)
    mm = ModelManager()
    rc = mm.get_relation_classifier()
    rc.test(X_test, Y_test)

if __name__ == '__main__':
    main()