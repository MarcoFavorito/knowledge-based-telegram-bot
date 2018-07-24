import utils.data as datautils
from operator import and_, or_
import constants as c
from exceptions import ConceptNotFound
from kbs.DBManager import DBManager, Item, Relation


class ConceptSelector():
    def __init__(self):
        self.db = DBManager()

    def retrieve_binary_answer(self, c1, c2, relation):
        s = self.db.session

        candidate_items = s.query(Item)\
            .filter(Item.relation==relation)\
            .filter(or_(Item.c1.like(c1+"::bn%"),Item.c1.like(c1)))\
            .filter(or_(Item.c2.like(c1+"::bn%"),Item.c2.like(c2))).all()

        return len(candidate_items)>0


    def get_right_concept(self, c1, relation):
        c2 = ""
        s = self.db.session

        candidate_items = s.query(Item) \
            .filter(Item.relation == relation) \
            .filter(or_(Item.c1.like(c1 + "::bn%"), Item.c1.like(c1))).all()

        if len(candidate_items)==0:
            if relation in c.INVERSE_RELATIONS:
                new_relation = c.INVERSE_RELATIONS[relation]
                candidate_items = s.query(Item) \
                    .filter(Item.relation == new_relation) \
                    .filter(or_(Item.c2.like(c1 + "::bn%"), Item.c2.like(c1))).all()
                if len(candidate_items)==0:
                    raise ConceptNotFound()
                else:
                    c2 = candidate_items[0].c1

        else:
            c2 =candidate_items[0].c2


        if not c2:
            raise ConceptNotFound()

        return datautils.clean_concept(c2)


