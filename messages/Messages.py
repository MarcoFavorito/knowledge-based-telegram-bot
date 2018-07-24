from enum import Enum

class Messages(Enum):
    START = "/start"
    WELCOME = "Hi!"
    ASK_DOMAIN_MSG = "What do you want to talk about?"


    QUERYING_MODE_STRING = "Yes"
    ENRICHING_MODE_STRING = "No"
    QE_SWITCH_MSG = "Do you want to ask a question?"

    Q_WELCOME_MSG = "Ok! Let me know your question."

    E_WELCOME_MSG = "Ok! Now I'll ask you a question"

    E_DUMMY_QUERY= "Who is the fairest of them all?"

    CONCEPT_RECOGNITION_FAILED = "Sorry, I didn't understand your message... speak simply, please."
    Q_CONCEPT_RETRIEVAL_FAILED = "I don't know, sorry"


    DOMAIN_NOT_VALID = "Sorry, the chosen domain is not valid. Please click one of the below buttons."
