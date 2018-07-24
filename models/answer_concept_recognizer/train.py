from models.answer_concept_recognizer.AnswerConceptRecognizerRNNBased import AnswerConceptRecognizerRNNBased
from models.answer_concept_recognizer import DATA_DIR, split_line, MODEL_PATH, read_file, DATA_DEV, DATA_TRAIN



def main():
    # X_train is a list of (question, answer, relation)
    # but it is an older version: you can ignore the question
    # only the answer and the relation are used


    X_train, Y_train = read_file(DATA_TRAIN)
    X_dev, Y_dev = read_file(DATA_DEV)

    acr = AnswerConceptRecognizerRNNBased()
    acr.make_vocab(map(lambda x: x[0], X_train))
    acr.make_vocab(map(lambda x: x[1], X_train))

    acr.train(X_train, Y_train, dev=(X_dev, Y_dev))

    acr.save(MODEL_PATH)



if __name__ == '__main__':
    main()