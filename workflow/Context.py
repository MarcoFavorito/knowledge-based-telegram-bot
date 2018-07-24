class Context(object):

    def __init__(self):
        self.domain = None
        self.candidate_relations = None
        self.relation2patterns = None
        self.query = None
        self.answer = None
        self.relation = None
        self.c1 = None
        self.c2 = None
        self.type2concepts = None
        self.last_id_inserted = -1
        pass

    def getCurrentDomain(self):
        return self.domain

    def getCandidateRelations(self):
        return self.candidate_relations

    def getRelation2Patterns(self):
        return self.relation2patterns

