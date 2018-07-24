import os
import sys
from sqlalchemy import Column, ForeignKey, Integer, Table, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy import create_engine



Base = declarative_base()

association_table = Table('domain2relation', Base.metadata,
    Column('domain_id', Integer, ForeignKey('domains.id')),
    Column('relation_id', Integer, ForeignKey('relations.id'))
)


class Domain(Base):
    __tablename__ = 'domains'

    id = Column(Integer, primary_key=True, autoincrement=True)
    simple_name = Column(String(250), unique=True)
    relations = relationship("Relation",
                             secondary=association_table,
                             backref=backref("relations", lazy="dynamic"))


class Relation(Base):
    __tablename__ = 'relations'

    id = Column(Integer, primary_key=True, autoincrement=True)
    simple_name = Column(String(250), unique=True)
    domains = relationship("Domain",
                           secondary=association_table,
                           backref=backref("domains", lazy="dynamic"))


class Pattern(Base):
    __tablename__ = 'patterns'

    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(String(250))
    relation_id = Column(Integer, ForeignKey("relations.id"))
    relation = relationship("Relation", foreign_keys=[relation_id])




class Item(Base):
    __tablename__ = 'items'

    _id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(String(250))
    answer = Column(String(250))
    relation = Column(String(250))
    context = Column(String(250))
    domains = Column(String(250))
    c1 = Column(String(250))
    c2 = Column(String(250))
    HASH = Column(Integer)

    # def __init__(self, question, answer, relation, context, domains, c1, c2, HASH, _id=-1):
    #     self._id = _id
    #     self.question = question
    #     self.answer = answer
    #     self.relation = relation
    #     self.context = context
    #     self.domains = domains
    #     self.c1 = c1
    #     self.c2 = c2
    #     self.HASH = HASH

    @classmethod
    def init_from_json(cls, json):
        question = json["question"].strip()
        answer = json["answer"].strip()
        relation = json["relation"].strip()
        context = json["context"].strip()
        domains = json["domains"][0].strip()
        c1 = json["c1"].strip()
        c2 = json["c2"].strip()
        HASH = json["HASH"]
        return Item(question=question,answer=answer,relation=relation,context=context,domains=domains,c1=c1,c2=c2,HASH=HASH)

    @classmethod
    def init_from_database_json(cls, json):
        _id = json["_id"]["$numberLong"]
        question = json["QUESTION"]
        answer = json["ANSWER"]
        relation = json["RELATION"]
        context = json["CONTEXT"]
        domains = json["DOMAIN"]
        c1 = json["C1"]
        c2 = json["C2"]
        HASH = json["HASH"]
        cls(question, answer, relation, context, domains, c1, c2, HASH, _id=_id)


    def to_json(self):
        json = {}
        json["_id"] = self._id
        json["question"] = self.question
        json["answer"] = self.answer
        json["relation"] = self.relation
        json["context"] = self.context
        json["domains"] = self.domains
        json["c1"] = self.c1
        json["c2"] = self.c2
        json["HASH"] = self.HASH
        return json

    @staticmethod
    def to_db_entry(json_entry):
        _id = int(json_entry["_id"]["$numberLong"])
        question = json_entry["QUESTION"]
        answer = json_entry["ANSWER"]
        relation = json_entry["RELATION"]
        context = json_entry["CONTEXT"]
        domains = json_entry["DOMAIN"]
        c1 = json_entry["C1"]
        c2 = json_entry["C2"]
        HASH = int(json_entry["HASH"])
        item = Item(
            _id=_id,
            question=question,
            answer=answer,
            relation=relation,
            context=context,
            domains=domains,
            c1=c1,
            c2=c2,
            HASH=HASH
        )
        return item

    def to_str(self):
        return "Item(question={0},answer={1},relation={2},context={3},domain={4},c1={5},c2={6}".format(
            repr(self.question), repr(self.answer), repr(self.relation), repr(self.context), repr(self.domains), repr(self.c1), repr(self.c2)
        )

    def to_repr(self):
        return " ".join([repr(self.question), repr(self.answer), repr(self.relation), repr(self.c1), repr(self.c2), repr(self.context), repr(self.domains)])

class Concept(Base):
    __tablename__= "concepts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    babelnet_id  = Column(String(250))
    mention = Column(String(250))
    bid_mention = Column(String(250), nullable=False)
    # domain_id = Column(Integer, ForeignKey("domains.id"))
    domain_simple_name = Column(Integer, ForeignKey("domains.simple_name"))
    domain = relationship("Domain", foreign_keys=[domain_simple_name])