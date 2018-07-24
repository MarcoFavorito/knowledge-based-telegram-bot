MAIN_DIR = "models/concept_recognizer"

DATA_DIR = MAIN_DIR+"/data"
DATA_TRAIN = DATA_DIR+"/train"
DATA_DEV = DATA_DIR+"/dev"
DATA_TEST = DATA_DIR+"/test"

MODEL_PATH = MAIN_DIR + "/model"


LSTM_SIZE = 128
EMBEDDING_SIZE = 300
DROPOUT = 0.05
BATCH_SIZE = 300
MAX_EPOCHS = 15
MAX_LENGTH = 35

def split_line(line):
    spl = line.strip().split("\t")
    try:
        return ((spl[0], spl[2]), spl[1].split(";"))
    except Exception:
        return []


def read_file(filepath):
    data = open(filepath, "r")
    lines = filter(lambda x: x, map(split_line, data.readlines()))

    X, Y = zip(*lines)
    return X, Y

def write_file(data, filepath):
    # here, data are items
    mod_data = list(map(lambda i: i.question.strip() +"\t" + i.c1.strip()+";"+i.c2.strip() + "\t" + i.relation, data))

    fout = open(filepath, "w")
    fout.write("\n".join([d.strip() for d in mod_data]))
