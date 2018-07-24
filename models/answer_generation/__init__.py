MAIN_DIR = "models/answer_generation"

DATA_DIR = MAIN_DIR+"/data"
DATA_TRAIN = DATA_DIR+"/train"
DATA_DEV = DATA_DIR+"/dev"
DATA_TEST = DATA_DIR+"/test"

MODEL_PATH = MAIN_DIR + "/model"


LSTM_SIZE = 200
EMBEDDING_SIZE = 200
DROPOUT = 0.0
BATCH_SIZE = 300
MAX_EPOCHS = 15
MAX_LENGTH = 35
EPOCHS = 1

def split_line(line):
    spl = line.strip().split("\t")
    # question, context
    return (spl[0], spl[1])


def read_file(filepath):
    data = open(filepath, "r")
    lines = map(split_line, data.readlines())

    X, Y = zip(*lines)
    return X, Y

def read_relations(filepath):
    data = open(filepath, "r")
    relations = list(map(lambda line:line.strip().split("\t")[5], data.readlines()))
    return relations


def write_file(data, filepath):
    # here, data are items
    mod_data = list(map(lambda i: "\t".join((i.question,i.context,i.answer,i.c1,i.c2,i.relation)), data))

    fout = open(filepath, "w")
    fout.write("\n".join([d.strip() for d in mod_data]))

