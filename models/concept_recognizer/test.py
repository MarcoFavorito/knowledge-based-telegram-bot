from models.ModelManager import ModelManager
from models.concept_recognizer import read_file, DATA_DIR, DATA_TEST, MODEL_PATH
from models.concept_recognizer.ConceptRecognizerRNNBased import ConceptRecognizerRNNBased


def main():
    X_test, Y_test = read_file(DATA_TEST)

    mm = ModelManager()
    new_ag = mm.get_concept_recognizer()
    new_ag.test(X_test, Y_test)

if __name__ == '__main__':
    main()