import os
import sys
from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy import func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session, sessionmaker
from sqlalchemy import create_engine
import config as conf
from kbs.ApiManager import ApiManager
from kbs.models import *
import json

class DBManager(object):
    __shared_state = {}

    def __init__(self, path=conf.DB_PATH):
        self.__dict__ = self.__shared_state
        if not hasattr(self, "session"):
            self.engine = create_engine('sqlite:///' + path)
            self.Base = Base
            Base.metadata.bind = self.engine
            session = sessionmaker()
            session.configure(bind=self.engine)
            self.session = Session()
    # and whatever else you want in your class -- that's all!

    def drop_all(self):
        Base.metadata.drop_all(self.engine)

    def create_all(self):
        Base.metadata.create_all(self.engine)

    def get_session(self):
        return self.session


    def sync(self, from_id=-1):
        """
        Syncronize with the central KB
        :param from_id:
        :return:
        """

        api_manager = ApiManager.init_from_conf(conf.API_CONF)

        # If we know from which id to start...
        if from_id!=-1:
            new_items = []
            items_num = api_manager.items_number_from(from_id)
            for id in range(from_id, from_id+items_num, 5000):
                new_items += api_manager.items_from(id)
                self.session.add_all(new_items)
            self.session.commit()
            return new_items

        # ... Otherwise, retrieve from the number of
        # item we have into the local mirror
        last_id = self.session.query(Item).count()
        last_id = 0 if last_id==None else last_id
        last_current_id = api_manager.items_number_from(0)

        # print("Current number of items: ", last_id)
        # print("number of items in the KB: ", last_current_id)

        tot_items = []
        if last_id==None or last_id<last_current_id:
            # Pagination
            for id in range(last_id, last_current_id, 5000):

                # print("current start id:",id)
                items = api_manager.items_from(id)

                # for i in items:
                #     print(i.to_repr())

                self.session.add_all(items)
                tot_items+=items
        self.session.commit()

        return tot_items


    def sync_from_file(self,
                       items_json_file=None,
                       patterns_file=None,
                       relations_file=None,
                       domains_file=None,
                       ):

        if items_json_file is not None:
            # Populate Items
            items = []
            with open(items_json_file, "r") as f:
                for l in f.readlines():
                    json_entry = json.loads(l.strip())
                    cur_item = Item.to_db_entry(json_entry)
                    items.append(cur_item)
                    # self.session.add(cur_item)
            self.session.add_all(items)
            self.session.commit()

        if relations_file is not None:
            #Populate Relations
            with open(relations_file) as f:
                for l in f.readlines():
                    cur_rel = Relation(simple_name=l.strip())
                    self.session.add(cur_rel)
            self.session.commit()

        if patterns_file is not None:
            #Populate Patterns
            with open(patterns_file, "r") as f:
                for l in f.readlines():
                    l = l.strip()
                    spl_l = l.split("\t")
                    relation_simple_name = spl_l[1]
                    cur_rel = self.session.query(Relation).filter(Relation.simple_name==relation_simple_name).one()
                    pattern = Pattern(
                        question=spl_l[0],
                        relation=cur_rel)
                    self.session.add(pattern)
            self.session.commit()

        if domains_file is not None:
            #Populate Domains
            with open(domains_file) as f:
                for l in f.readlines():
                    l = l.strip()
                    spl_l = l.split("\t")
                    domain_simple_name = (spl_l[0]).strip()
                    relation_simple_names = (spl_l[1:])
                    cur_rels = self.session.query(Relation).filter(Relation.simple_name.in_(relation_simple_names)).all()
                    cur_dom = Domain(
                        simple_name=domain_simple_name,
                    )
                    for r in cur_rels:
                        cur_dom.relations.append(r)
                    self.session.add(cur_dom)
            self.session.commit()




