from models.answer_concept_recognizer import read_file, DATA_DIR, DATA_TEST, MODEL_PATH
from models.answer_concept_recognizer.AnswerConceptRecognizerRNNBased import AnswerConceptRecognizerRNNBased


def main():
    X_test, Y_test = read_file(DATA_TEST)
    new_ag = AnswerConceptRecognizerRNNBased.load(MODEL_PATH)
    new_ag.test(X_test, Y_test)

if __name__ == '__main__':
    main()