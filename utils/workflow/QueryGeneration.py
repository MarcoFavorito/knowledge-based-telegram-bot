from kbs.DBManager import DBManager, Item, Relation, Domain, Pattern
from kbs.models import Concept
from sqlalchemy import or_
import random
import re
import constants as c

"""query for retrieve the least used concept"""
GET_CONCEPT_OCCURRENCES = """select concept_name, sum(cc) as total_cc, dmn from 
(
select distinct bid_mention as concept_name, 0 as cc, domain_simple_name as dmn from concepts as cpt 
union all 
select i.c1 as concept_name, count(*) as cc, i.domains as dmn from items as i where dmn ="{0}" group by  concept_name
union ALL 
select i.c2 as concept_name, count(*) as cc, i.domains as dmn from items as i where dmn="{0}" group by  concept_name
)
where dmn = "{0}"
group by concept_name order by  total_cc asc

"""


class QueryGeneration(object):
    def __init__(self):
        self.db = DBManager()
        pass


    def get_concept_by_domain(self, domain_simple_name):
        """retrieve the concept with least usage"""
        res = self.db.engine.execute(GET_CONCEPT_OCCURRENCES.format(domain_simple_name)).fetchone()
        return res[0]

    def get_relation_by_concept(self, concept_mention, domain_simple_name):
        """retrieve the least used relation for that concept"""

        chosen_domain = self.db.session.query(Domain).filter(Domain.simple_name==domain_simple_name).one()
        candidate_relations = self.db.session.query(Relation).filter(Relation.domains.contains(chosen_domain)).all()
        items = self.db.session.query(Item)\
            .filter(or_(Item.c1.like(concept_mention + "%"), Item.c2.like(concept_mention + "%")))\
            .all()

        candidate_relations = [r.simple_name for r in candidate_relations]
        items = filter(lambda x: x.relation in candidate_relations, items)

        rel_counts = dict(zip(candidate_relations, [0]*len(candidate_relations)))
        for i in items:
            if not i.relation in rel_counts:
                rel_counts[i.relation]=0
            rel_counts[i.relation]+=1

        print(rel_counts)
        rel_counts = filter(lambda x: x[0] in candidate_relations,rel_counts.items())
        rel_counts = sorted(rel_counts, key=lambda x: x[1])

        assert len(rel_counts)>0
        chosen_relation = rel_counts[0][0]
        return chosen_relation


    def generate_query(self, concept, relation):
        """retrieve a random question pattern by relation and then substitute the left concept"""

        chosen_relation = self.db.session.query(Relation).filter(Relation.simple_name==relation).one()
        patterns = self.db.session.query(Pattern).filter(Pattern.relation==chosen_relation).all()
        patterns_strings = [p.question for p in patterns]

        non_binary_patterns = filter(
            lambda x: re.search(c.CONCEPT_PATTERN.format("X"), x) and not
                      re.search(c.CONCEPT_PATTERN.format("Y"), x), patterns_strings)

        chosen_pattern = random.choice(list(non_binary_patterns))

        query = chosen_pattern.replace("X", concept)

        return query




