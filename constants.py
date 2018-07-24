# Some constants
NULL_BABELNET_ID = "NULL_BID"
NULL_TYPE = "NULL_T"
CUSTOM_TYPE = "CUSTOM_TYPE"

# POS tags
VERB_POSTAG = "VERB"
PROPN_POSTAG = "PROPN"
PRON_POSTAG = "PRON"
NOUN_POSTAG = "NOUN"

# dep tags
NSUBJ_DEPTAG = "nsubj"
NSUBJPASS_DEPTAG = "nsubjpass"
POBJ_DEPTAG = "pobj"
COBJ_DEPTAG = "cobj"
DOBJ_DEPTAG = "dobj"
ATTR_DEPTAG = "attr"

SPECIAL_DEP_TAGS = [NSUBJ_DEPTAG, NSUBJPASS_DEPTAG, POBJ_DEPTAG, ATTR_DEPTAG, COBJ_DEPTAG, DOBJ_DEPTAG]
SUBJ_DEPTAGS = [NSUBJ_DEPTAG, NSUBJPASS_DEPTAG]
# OBJ_DEPTAGS = [POBJ_DEPTAG, ATTR_DEPTAG, COBJ_DEPTAG, DOBJ_DEPTAG]
OBJ_DEPTAGS = [POBJ_DEPTAG, COBJ_DEPTAG, DOBJ_DEPTAG]

#Used in utils.misc.my_tokenizer
ENT_TAG = "_ENT_"
NUM_TAG = "_NUM_TAG_"
PROPN_TAG = "_PROPN_"


#Used in entities
LEFT_CONCEPT_TAG = "xxx"
RIGHT_CONCEPT_TAG = "yyy"

N_ENT_TAG = "n"
RIGHT_ENT_TAG = "r"
LEFT_ENT_TAG = "l"
Y_ENT_TAG = "y"


PAD_TAG = "pad"
UNK_TAG = "unk"
NIL_TAG = "nil"


deptags = ["nil", "nsubj", "nsubjpass", "pobj", "cobj", "dobj", "attr"]
postags = ["nil","VERB", "PROPN", "PRON","NOUN"]
tbtags = ["nil"]

entity_tags = [NIL_TAG,LEFT_ENT_TAG, RIGHT_ENT_TAG, N_ENT_TAG]

CONCEPT_PATTERN = "(?<=\W){0}$|^{0}(?=\W)|(?<=\W){0}(?=\W)|^{0}$"

RELATIONS = ["SIMILARITY", "SHAPE", "ACTIVITY", "SOUND", "SIZE", "GENERALIZATION", "PART", "PLACE", "SMELL", "TIME", "TASTE", "COLOR", "MATERIAL", "SPECIALIZATION", "HOW_TO_USE", "PURPOSE"]

old2new = {'activity': 'ACTIVITY', 'colorPattern': 'COLOR', 'generalization': 'GENERALIZATION', 'howToUse': 'HOW_TO_USE', 'material': 'MATERIAL', 'part': 'PART', 'place': 'PLACE', 'purpose': 'PURPOSE', 'shape': 'SHAPE', 'similarity': 'SIMILARITY', 'size': 'SIZE', 'smell': 'SMELL', 'sound': 'SOUND', 'specialization': 'SPECIALIZATION', 'taste': 'TASTE', 'time': 'TIME'}

INVERSE_RELATIONS = {"SPECIALIZATION":"GENERALIZATION"}