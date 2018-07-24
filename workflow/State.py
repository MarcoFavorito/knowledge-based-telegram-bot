from enum import Enum
#Enum for the possible states of the request
class State(Enum):
    START = 1
    ASK_DOMAIN = 2
    RECEIVE_DOMAIN = 9
    QE_ASK_SWITCH = 3
    QE_SWITCH = 4

    Q_PROCESSING = 5
    Q_ANSWER_GENERATION = 10

    E_QUERY_GENERATION = 6
    E_ANSWER_PROCESSING = 7
    FINISH_WORKFLOW = 8
