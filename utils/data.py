import re
from sqlalchemy.orm import load_only
import constants as c
from kbs.DBManager import DBManager, Item
from unidecode import unidecode
import numpy as np
import utils.graph_utils as gut
from utils.dependency_parser import get_spacy_parser


def get_items_from_db(max_per_relation=-1, filtering=True):
    db = DBManager()
    import random
    items = db.session.query(Item).options(load_only("_id","question", "relation", "c1", "c2", "context")).__iter__()

    filtered_items = items
    if filtering:
        filtered_items = filter(filter_item, items)

    items_by_relation = {}

    for i in filtered_items:
        if i.relation not in items_by_relation: items_by_relation[i.relation]=[]
        items_by_relation[i.relation].append(i)

    # relation_counts = [(r, len(l)) for r, l in items_by_relation.items()]
    # print(sum([c for _, c in relation_counts]))
    # print(relation_counts)

    final_items = []
    for r, item_list in items_by_relation.items():
        if max_per_relation==-1:
            final_items+=item_list
        else:
            k = len(item_list) if len(item_list)<max_per_relation else max_per_relation
            final_items+= random.sample(item_list, k)

    return final_items




def build_train_relclass():
    items = get_items_from_db(max_per_relation=1000)

    y = list(map(lambda x: x.relation, items))
    X = list(map(lambda x: x.question, items))

    return X, y




def filter_item(item):
    """
    :param item: an Item obj (see kbs.models)
    filter item if:
        - the concept is not in the form: (concept_mention:babelID) OR (concept_mention)
        - the left concept is not into the question field.
    """
    c1 = item.c1
    c2 = item.c2
    q = item.question
    ctx = item.context

    if not filter_concept(c1) or not filter_concept(c2):
        # print("concept problem:",item.relation, q, c1, c2)
        return False

    # Extract c1 name
    c1_name = c1.split("::")
    try:
        assert len(c1_name) <= 2;
    except Exception:
        # print("C1 problem:", item.relation, q, c1, c2)
        return False
    c1_name = c1_name[0].strip()

    # Extract c2 name
    c2_name = c2.split("::")
    try:
        assert len(c2_name) <= 2;
    except Exception:
        # print("C2 problem:", item.relation, q, c1, c2)
        return False
    c2_name = c2_name[0].strip()

    if c1_name not in q:
        # print("QUESTION problem:", item.relation, q, c1, c2)
        return False

    # if c1_name not in ctx or c2_name not in ctx:
    #     return False

    return True

def filter_item_with_context(items):
    """
    filter item if the concept mentions does not appears in the question, in the answer of in the context.
    If yes, replace with the concept tags (xxx or yyy).
    :param items: a list of Item objs (see kbs.models.py)
    :return: list of modified items
    """

    new_items = []
    for item in items:
        q = clean_question(item.question)
        ans = clean_question(item.answer)
        ctx = item.context.strip()
        c1_name = clean_concept(item.c1)
        c2_name = clean_concept(item.c2)





        # print("=" * 50)
        # print("=" * 50)
        # print(repr(q), repr(ctx), repr(c1_name), repr(c2_name))

        pattern = c.CONCEPT_PATTERN
        escaped_c1 = re.escape(c1_name)
        escaped_c2 = re.escape(c2_name)
        c1_pattern = pattern.format(escaped_c1)
        c2_pattern = pattern.format(escaped_c2)

        #question
        # print("question")
        c1_mentions = list(re.finditer(c1_pattern, q))
        c1_ranges = [r.span() for r in c1_mentions]

        try:
            assert len(c1_ranges) != 0
            # assert ctx[c1_idx:c1_idx + len(c1_name)] == c1_name
        except Exception:
            continue

        # print((question[:c1_idx] + "#"*len(c1) + question[c1_idx+len(c1):]))
        c1_idx = c1_ranges[0][0]
        q_without_c1 =q[:c1_idx] + c.LEFT_CONCEPT_TAG + q[c1_idx + len(c1_name):]
        # print(q_without_c1)

        c2_mentions = list(re.finditer(c2_pattern, q_without_c1))
        c2_ranges = [r.span() for r in c2_mentions]
        if len(c2_ranges)!=0:
            c2_idx = c2_ranges[0][0]
            q_without_c1c2 = q_without_c1[:c2_idx] + c.RIGHT_CONCEPT_TAG + q_without_c1[c2_idx + len(c2_name):]
            # print(q_without_c1c2)
            # print("NO!")
            continue


        # ----------------------------------------------------------------------------------------------------


        # context
        # print("context")
        c1_mentions = list(re.finditer(c1_pattern, ctx))
        c1_ranges = [r.span() for r in c1_mentions]

        try:
            assert len(c1_ranges) != 0
            # assert ctx[c1_idx:c1_idx + len(c1_name)] == c1_name
        except Exception:
            # print(c1_ranges, "c1r len!=0")
            continue
        c1_idx = c1_ranges[0][0]
        # print((question[:c1_idx] + "#"*len(c1) + question[c1_idx+len(c1):]))

        ctx_without_c1 = ctx[:c1_idx] + c.LEFT_CONCEPT_TAG + ctx[c1_idx + len(c1_name):]
        c2_mentions = list(re.finditer(c2_pattern, ctx_without_c1))
        c2_ranges = [r.span() for r in c2_mentions]
        try:
            assert len(c2_ranges) != 0
            # assert ctx[c1_idx:c1_idx + len(c1_name)] == c1_name
        except Exception:
            # print(c2_ranges, "c2r len!=0")
            continue
        c2_idx = c2_ranges[0][0]
        ctx_without_c1c2 = ctx_without_c1[:c2_idx] + c.RIGHT_CONCEPT_TAG + ctx_without_c1[c2_idx + len(c2_name):]

        # print(ctx_without_c1)
        # print(ctx_without_c1c2)

        #----------------------------------------------------------------------------------------------------
        # answer

        c1_pattern = pattern.format(escaped_c1.lower())
        c2_pattern = pattern.format(escaped_c2.lower())

        c2_mentions = list(re.finditer(c2_pattern, ans.lower()))
        c2_ranges = [r.span() for r in c2_mentions]

        try:
            assert len(c2_ranges) != 0
        except:
            # print(repr(ans), repr(c1_name), repr(c2_name), c2_pattern)
            continue
        c2_idx = c2_ranges[0][0]
        a_without_c2 = ans[:c2_idx] + c.RIGHT_CONCEPT_TAG + ans[c2_idx + len(c2_name):]
        a_without_c2 = a_without_c2.strip()
        # print(a_without_c1c2)
        # print("NO!")

        c1_mentions = list(re.finditer(c1_pattern, a_without_c2))
        c1_ranges = [r.span() for r in c1_mentions]
        if len(c1_ranges) != 0:
            # print((question[:c1_idx] + "#"*len(c1) + question[c1_idx+len(c1):]))
            c1_idx = c1_ranges[0][0]
            a_without_c1c2 = a_without_c2[:c1_idx] + c.LEFT_CONCEPT_TAG + a_without_c2[c1_idx + len(c1_name):]
            # print(q_without_c1)
        else:
            a_without_c1c2 = a_without_c2

        #-------------------------------------------------------------------------------------------------

        new_items.append(
            Item(
                question=q_without_c1,
                context=ctx_without_c1c2,
                answer=a_without_c1c2,
                c1=c1_name,
                c2=c2_name,
                relation=item.relation
            )
        )
    return new_items


def filter_concept(c):
    # Check if concept field is not compliant with the pattern: ".+(::bn:[0-9]{8}[a-z])?"
    import re
    # concept_pattern = "^.+(::bn:[0-9]{8}[a-z])?$"
    negative_pattern = "^bn:[0-9]*"
    if re.search(negative_pattern, c):
        return False

    negative_pattern = "bn:[0-9]*"
    if len(re.findall(negative_pattern, c)) > 1:
        return False

    negative_pattern = "::bn"
    if len(re.findall(negative_pattern, c)) > 1:
        return False

    return True

def clean_concept(c):
    cc = unidecode(c.split("::")[0].strip())
    return cc

def clean_question(question):
    cq = unidecode(re.sub(" *\?$", "", question)).strip()
    return cq

def get_features_from_spacy_token(t):
    try:
        postag = c.postags.index(t.pos_)
    except Exception:
        postag = 0

    try:
        tbtag = c.tbtags.index(t.tag_)
    except Exception:
        tbtag = 0

    try:
        deptag = c.deptags.index(t.dep_)
    except Exception:
        deptag = 0

    return postag, tbtag, deptag


def to_ohenc(num, n_max):
    arr = np.zeros(n_max)
    arr[num] = 1
    return arr


def is_conflicting_range(range_in, range_out):
    return (
        range_in[1] >= range_out[0] and range_in[1] <= range_out[1]
         or
        range_in[0] >= range_out[0] and range_in[0] <= range_out[1]
        or
        range_in[0]<=range_out[0] and range_in[1] >= range_out[1]
    )




def simplify_ctx(ctx):
    """dep-parse the context and find the shortest path between xxx and yyy;
    then enrich the path with auxiliary words and others"""
    parser = get_spacy_parser()
    # print("="*50)
    # print(orig_ctx)
    # print(ctx)

    doc = parser(ctx)
    sents = list(doc.sents)
    try:
        assert len(sents) == 1
    except AssertionError:
        c1_idx = ctx.find(c.LEFT_CONCEPT_TAG)
        c2_idx = ctx.find(c.RIGHT_CONCEPT_TAG)
        sent = ctx[c1_idx: c2_idx+len(c.RIGHT_CONCEPT_TAG)]
        if sent=="":
            sent = ctx[c2_idx: c1_idx + len(c.LEFT_CONCEPT_TAG)]
        doc = parser(sent)
        sents = list(doc.sents)
        try:
            assert len(sents)==1
        except:
            return ""

    sent = sents[0]
    xxx_tok = None
    yyy_tok = None

    for t in sent:
        if (c.LEFT_CONCEPT_TAG in t.text):
            xxx_tok = t
            # print("xxx:", xxx_tok.text)
            break

    for t in sent:
        if (c.RIGHT_CONCEPT_TAG in t.text):
            yyy_tok = t
            # print("yyy:", yyy_tok.text)
            break

    assert xxx_tok and yyy_tok

    if xxx_tok.i < yyy_tok.i:
        start = xxx_tok
        end = yyy_tok
    else:
        start =yyy_tok
        end = xxx_tok

    G = gut.build_networkXGraph_from_spaCy_depGraph(sent)
    sh_path = gut.shortest_path(G, source=start, target=end)
    # print("Shortest path: ", sh_path)
    final_string = ""
    left_string = ""
    right_string = ""
    added_tokens = []
    for t in sh_path:
        left_children = t.lefts
        right_children = t.lefts
        for child in left_children:
            if child in sh_path or child in added_tokens: continue
            if child.pos_=="DET" or (t.pos_=="VERB" and child.pos_=="VERB"):
                left_string+=" " + child.text
                added_tokens.append(child)
        for child in right_children:
            if child in sh_path or child in added_tokens: continue
            if child.pos_ == "DET" or (t.pos_ == "VERB" and child.pos_ == "VERB"):
                right_string += " " + child.text
                added_tokens.append(child)


        final_string += left_string + " " + t.text + right_string
        # print(repr(final_string), repr(left_string), repr(t.text), repr(right_string))
        left_string = ""
        right_string = ""

    res = final_string.strip() + " ."
    # res = build_string()
    # print(res)
    return res

def filter_item_by_answer(item):
    """filter items if in the answer does not appear the right concept"""
    q = clean_question(item.question)
    ans = clean_question(item.answer)
    c1_name = clean_concept(item.c1)
    c2_name = clean_concept(item.c2)

    pattern = c.CONCEPT_PATTERN
    escaped_c1 = re.escape(c1_name)
    escaped_c2 = re.escape(c2_name)
    c1_pattern = pattern.format(escaped_c1)
    c2_pattern = pattern.format(escaped_c2)

    # question
    # print("question")
    c1_mentions = list(re.finditer(c1_pattern, q))
    c1_ranges = [r.span() for r in c1_mentions]

    try:
        assert len(c1_ranges) != 0
        # assert ctx[c1_idx:c1_idx + len(c1_name)] == c1_name
    except Exception:
        return False


    # answer
    c1_pattern = pattern.format(escaped_c1.lower())
    c2_pattern = pattern.format(escaped_c2.lower())

    c2_mentions = list(re.finditer(c2_pattern, ans.lower()))
    c2_ranges = [r.span() for r in c2_mentions]

    try:
        assert len(c2_ranges) != 0
    except:
        # print(repr(ans), repr(c1_name), repr(c2_name), c2_pattern)
        return False

    return True
