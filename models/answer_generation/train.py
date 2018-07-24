from models.answer_generation import DATA_DIR, split_line, MODEL_PATH, read_file, DATA_DEV, DATA_TRAIN
from models.answer_generation.AnswerGenerator import AnswerGenerator




def main():

    X_train, Y_train = read_file(DATA_TRAIN)
    # X_train: list of modified question (i.e. concept tags instead of concept mentions)
    # Y_train: list of modified context (i.e. concept tags instead of concept mentions)

    ag = AnswerGenerator()
    ag.train(X_train, Y_train)
    ag.save(MODEL_PATH)


if __name__ == '__main__':
    main()