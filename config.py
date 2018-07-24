# VERBOSE = False
# VERBOSITY = 2

# Telegram token - if you want to run the bot on Telegram
TOKEN = "your-telegram-token"
# BabelNet key - optional
BN_KEY = "your-babelnet-key"


PROT = "http"
HOST = "151.100.179.26"
PORT = "8080"
PATH = "/KnowledgeBaseServer/rest-api/"

DB_PATH = "kbs/kb_mirror.db"


API_CONF = {
    "key":BN_KEY,
    "prot":PROT,
    "host":HOST,
    "port":PORT,
    "path":PATH,
}


KBS_DUMP_PATH= "kbs/data/db_chatbot.dump.txt"
RELATIONS_PATH = "kbs/data/relations"
DOMAINS2RELATIONS_OLD_PATH = "kbs/data/chatbot_maps/domains_to_relations.tsv"
DOMAINS2RELATIONS_NEW_PATH = "kbs/data/domains_to_relations.tsv"
PATTERNS2RELATIONS_PATH = "kbs/data/cleaned_patterns.tsv"
PATTERNS_FOLDER_PATH = "kbs/data/patterns/"

CONCEPT2DOMAIN_WIKI = "kbs/data/chatbot_maps/BabelDomains_full/BabelDomains/babeldomains_wiki.txt"
CONCEPT2DOMAIN_BABELNET = "kbs/data/chatbot_maps/BabelDomains_full/BabelDomains/babeldomains_wordnet.txt"


RELATION_CLASSIFIER_MODEL = "models/relation_classifier/model"
CONCEPT_RECOGNIZER_MODEL = "models/concept_recognizer/model"
ANSWER_GENERATION_MODEL = "models/answer_generation/model"
ANSWER_CONCEPT_RECOGNIZER_MODEL = "models/answer_concept_recognizer/model"
