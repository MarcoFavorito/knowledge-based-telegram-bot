import time

import math

import config
import constants as c
from nltk import Tree
import re


def log_print(*args, **kwargs):
    """
	A simple wrapper of the built-in print function.
	It prints only if configuration.VERBOSE is true
	"""
    if "verbosity" in kwargs:
        verbosity = kwargs["verbosity"]
        del kwargs["verbosity"]
    else:
        verbosity = 1
    if (config.VERBOSE and verbosity <= config.VERBOSITY):
        print(*args, **kwargs)


def tok_format(tok):
    """
	Make a summary string from a spaCy token, containing:
	 tok.orth_, which is the word[s];
	 tok.ent_id_, which is the entity id,
	 tok.dep_, which is the dependency tag
	"""
    return "_".join([tok.orth_, tok.tag_, tok.pos_, tok.dep_])


def to_nltk_tree(node):
    """
	Returns a nltk.Tree object from a spaCy dependency graph.
	It should be calld with node set as the root node of the dependency graph.
	"""
    if node.n_lefts + node.n_rights > 0:
        return Tree(tok_format(node), [to_nltk_tree(child) for child in node.children])
    else:
        return Tree(tok_format(node), [])


def clean_sentence(sentence):
    """
	Clean a sentence from some useless stuff (brackets, quotation marks etc.)
	:param sentence: a string, representing a sentence.
	:return: the same pair of (sentence, disambiguations) without the tokens relative to bad substrings.
	"""

    # regex that solves out problem
    # r = re.compile(" ?(\(|\[) .+? (\)|\])| ?``| ?''|,|–,"# )
    r = re.compile("[,\]\[\(\)\–\—]")

    new_sentence = r.sub("", sentence)

    new_sentence = re.sub("\\/", " ", new_sentence)
    new_sentence = re.sub(" +", " ", new_sentence)
    return new_sentence.lower()


def transform_word_custom(word):
    """
	This function has no claim to be "complete",
	but I noticed a slight improvement for the self-learned
	vocabulary, since it eliminates some wasteful redundancies.
	(Examples:
		- "09:00", "01/01/2000" and other time format to "##:##", "##/##/####"
		- http://.* in http://website
		- Every character repeated more than three consecutive times is normalized to three times
		 (Trying to catch mispelled words or not regular punctiation and collapse them).
	:param word: the word to modify.
	:return: custom_word, the modified word.
	"""
    custom_word = word
    # if re.search(r"^(http://|www\.).+", custom_word):
    # 	custom_word = re.sub(r"^(http://|www\.).+", r"\1website", custom_word)
    # elif re.search(r"^.+@.+\.\w+", word):
    # 	custom_word = re.sub(r"^.+@.+\.\w+", r"email@domain.name", custom_word)

    if re.search(r"[0-9]+", custom_word):
        custom_word = re.sub("[0-9]+", c.NUM_TAG, custom_word)
    # custom_word = re.sub("#num#{6,}", r"#num#"*5, custom_word)
    # elif re.search(r"(.)\1{4,}", word):
    # 	custom_word = re.sub(r"(.)\1{4,}", r"\1\1\1", custom_word)

    return custom_word


def merge_concept_tags(words, tags):
    """
    given a list of words and a concept tags to them,
    merge the contiguous sequences of the same tags.
    :return: a dictionary:
            {concept_tag : list of tokenized concept mentions}

            e.g.
            Is the University of Rome in Rome?
            {
                "l": [ ["University", "of", "Rome"] ],
                "r": [ ["Rome"] ]
             }
    """

    left_indexes = [i for i, t in enumerate(tags) if t == c.LEFT_ENT_TAG]
    right_indexes = [i for i, t in enumerate(tags) if t == c.RIGHT_ENT_TAG]

    # get ranges made by continuous number sequences
    left_ranges = get_ranges_from_indexes(left_indexes)
    right_ranges = get_ranges_from_indexes(right_indexes)

    # print(left_ranges)
    # print(right_ranges)

    type2conceptList = {c.LEFT_ENT_TAG:[], c.RIGHT_ENT_TAG:[]}

    # From ranges, retrieve tokens
    for r in left_ranges:
        c1 = words[r[0]:r[1]]
        type2conceptList[c.LEFT_ENT_TAG].append(c1)

    for r in right_ranges:
        c2 = words[r[0]:r[1]]
        type2conceptList[c.RIGHT_ENT_TAG].append(c2)

    return type2conceptList


def get_ranges_from_indexes(indexes):
    if len(indexes) == 0: return []
    ranges = []
    sorted_indexes = sorted(indexes)
    cur_min = min(sorted_indexes)
    cur_max = cur_min
    for idx in indexes:
        if idx == cur_max:
            cur_max = idx + 1
        else:
            ranges.append((cur_min, cur_max))
            cur_min = idx
            cur_max = idx + 1
    if cur_max - cur_min >= 1:
        ranges.append((cur_min, cur_max))
    return ranges


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def extract_concepts(type2concepts, from_question=True):
    """
    :param type2concepts: {concept_tag : list of tokenized concept mentions}
            e.g.
            Is the University of Rome in Rome?
            {
                "l": [ ["University", "of", "Rome"] ],
                "r": [ ["Rome"] ]
             }
             where "l" and "r" stand for "left" and "right" concept
    :param from_question: if the concepts come from a question or an answer
    :return: (c1, c2), the concept strings for the left and right concept
    """
    c1_list = type2concepts[c.LEFT_ENT_TAG]
    c2_list = type2concepts[c.RIGHT_ENT_TAG]
    # print(c1_list, c2_list)

    if len(c1_list)==1:
        c1 = c1_list[0]
    elif len(c1_list)==0:
        c1 = ""
    else:
        # if there is more than one candidate concept, sort by concept length
        c1_list = sorted(enumerate(c1_list), key=lambda x: (len(x[1]),x[0]))
        # print("sorted c_list:", c1_list)
        # pick the longest
        c1 = c1_list[0][1]

    if len(c2_list)==1:
        c2 = c2_list[0]
    elif len(c2_list)==0:
        c2 = ""
    else:
        # if there is more than one candidate concept, sort by concept length
        c2_list = sorted(enumerate(c2_list), key=lambda x: (len(x[1]), x[0]))
        # print("sorted c_list:", c2_list)
        # pick the longest
        c2 = c2_list[0][1]

    if from_question:
        if not c1 and c2:
            c1=c2
            c2=""
            # print("Switch concepts: it is a question")
            # print(c1,c2)
    else:
        if not c2 and c1:
            c1 = c2
            c2 = ""
            # print(c1, c2)
            # print("Switch concepts: it is an answer")
    return " ".join(c1), " ".join(c2)


def rectify_answer(generated_answer):
    """remove duplicate consecutive words
    e.g. My My Name Name -> My Name"""
    words = generated_answer.split()
    new_words = [words[0]]
    for w in words[1:]:
        if new_words[-1]!=w:
            new_words.append(w)

    return " ".join(new_words)