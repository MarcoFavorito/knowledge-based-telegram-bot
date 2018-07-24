import config
from kbs.DBManager import DBManager
import utils.data as datautils
from kbs.models import Concept, Domain


def insert_from_items(db):
    items = datautils.get_items_from_db()
    new_concepts = []
    bid_mentions = set(map(lambda x: x[0], db.session.query(Concept.mention).__iter__()))

    for i in items:
        if datautils.filter_item(i):
            (c1_mention, c1_babelnet_id) = i.c1.split("::") if len(i.c1.split("::")) == 2 else (i.c1, "")
            (c2_mention, c2_babelnet_id) = i.c2.split("::") if len(i.c2.split("::")) == 2 else (i.c2, "")

            new_c1 = Concept(
                babelnet_id=c1_babelnet_id,
                mention=c1_mention,
                bid_mention=i.c1
            )
            new_c2 = Concept(
                babelnet_id=c2_babelnet_id,
                mention=c2_mention,
                bid_mention=i.c2
            )

            if not new_c1.bid_mention in bid_mentions:
                new_concepts.append(new_c1)
                bid_mentions.add(new_c1.bid_mention)

            if not new_c2.bid_mention in bid_mentions:
                new_concepts.append(new_c2)
                bid_mentions.add(new_c2.bid_mention)

    db.session.add_all(new_concepts)
    db.session.commit()

def insert_from_babeldomains_wiki(db):
    domains = db.session.query(Domain).all()
    domain_name_to_domain_obj = {d.simple_name: d for d in domains}

    bid_mentions = set(map(lambda x: x[0], db.session.query(Concept.mention).__iter__()))

    BUFFER_SIZE = 1000000
    i = 0
    new_concepts = []
    babeldomains_wiki = map(lambda x: x.split("\t")[:2], open(config.CONCEPT2DOMAIN_WIKI).readlines())

    for mention, domain_name in babeldomains_wiki:
        if mention in bid_mentions:
            continue
        bid_mentions.add(mention)
        new_c = Concept(mention=mention, bid_mention=mention, domain=domain_name_to_domain_obj[domain_name])
        new_concepts.append(new_c)

        if len(new_concepts)>=BUFFER_SIZE:
            print(i)
            i+=1
            db.session.add_all(new_concepts)
            db.session.commit()
            new_concepts = []

    db.session.add_all(new_concepts)
    db.session.commit()


def main():
    db = DBManager()
    Concept.__table__.drop()
    Concept.__table__.create()

    insert_from_items(db)
    insert_from_babeldomains_wiki(db)

"""
select concept_name, sum(cc) as total_cc from 
(
select distinct bid_mention as concept_name, 0 as cc from concepts as cpt 
union all 
select i.c1 as concept_name, count(*) as cc from items as i where domain="%s" group by  concept_name
union ALL 
select i.c2 as concept_name, count(*) as cc from items as i where domain="%s" group by  concept_name
)
group by concept_name order by  total_cc asc

"""





if __name__ == '__main__':

    main()