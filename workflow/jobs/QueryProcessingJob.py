from kbs.DBManager import DBManager, Domain, Relation
from models.ModelManager import ModelManager
from models.concept_recognizer.ConceptRecognizerRNNBased import ConceptRecognizerRNNBased
from models.relation_classifier.RelationClassifierRNNBased import RelationClassifierRNNBased
from workflow.Job import Job
from workflow.State import State
import numpy as np
import config
import gc
import tensorflow as tf

class QueryProcessingJob(Job):
    """Process the user query
    1) predict the relation
    2) predict the concepts
    3) Schedule the QueryAnswerGenerationJob
    """

    def validate_input(self, input_text) -> bool:
        return True

    def the_job(self, input_text):
        query = input_text
        context = self.workflow_manager.getContext()
        chosen_domain = context.domain

        db = DBManager()
        model_manager = ModelManager()

        # Relation Prediction

        # retrieve candidate relations (i.e. filtering by domain)
        chosen_domain = db.session.query(Domain).filter(Domain.simple_name == chosen_domain).one()
        candidate_relations = db.session.query(Relation).filter(Relation.domains.contains(chosen_domain)).all()
        candidate_relations = [r.simple_name for r in candidate_relations]

        # load the model and predict the relation
        relation_classifier = model_manager.get_relation_classifier()
        relation = relation_classifier.predict(query, candidate_relations=candidate_relations)
        self.joblog("The predicted relation is: " + relation)


        #Concept recognition
        concept_recognizer = model_manager.get_concept_recognizer()
        # type2concepts is
        type2concepts = concept_recognizer.predict(query, relation)
        self.joblog("Predicted concepts are: " + str(type2concepts))

        # Save in Context of WorkflowManager
        self.joblog("Saving in the context the query: %s" % input_text)
        context.query = query
        context.relation = relation
        context.type2concepts = type2concepts



    def get_new_state(self):
        return self.workflow_manager._dispatch(State.Q_ANSWER_GENERATION, "")

