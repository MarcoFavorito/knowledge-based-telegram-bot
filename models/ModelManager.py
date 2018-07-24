import tensorflow as tf
import constants as c
import config
from models.answer_concept_recognizer.AnswerConceptRecognizerRNNBased import AnswerConceptRecognizerRNNBased
from models.answer_generation.AnswerGenerator import AnswerGenerator
from models.concept_recognizer.ConceptRecognizerRNNBased import ConceptRecognizerRNNBased
from models.relation_classifier.RelationClassifierRNNBased import RelationClassifierRNNBased


class ModelManager():
    """
    A singleton class for manage all the models
    with the Borg design pattern
    ... Thanks, Alex Martella!
    """
    __shared_state = {}

    def __init__(self):
        self.__dict__ = self.__shared_state
        if not hasattr(self, "relation_classifier") and \
           not hasattr(self, "concept_recognizer") and \
           not hasattr(self, "answer_generator"):
            # Assume that you have 12GB of GPU memory and want to allocate ~4GB:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

            self.relation_classifier = None
            self.concept_recognizer = None
            self.answer_generator = None
            self.answer_concept_recognizer = None


    def get_relation_classifier(self):
        if not self.relation_classifier:
            self.relation_classifier = RelationClassifierRNNBased.load(config.RELATION_CLASSIFIER_MODEL)
        return self.relation_classifier

    def get_concept_recognizer(self):
        if not self.concept_recognizer:
            self.concept_recognizer = ConceptRecognizerRNNBased.load(config.CONCEPT_RECOGNIZER_MODEL)
        return self.concept_recognizer

    def get_answer_generator(self):
        if not self.answer_generator:
            self.answer_generator = AnswerGenerator.load(config.ANSWER_GENERATION_MODEL)
        return self.answer_generator

    def get_answer_concept_recognizer(self):
        if not self.answer_concept_recognizer:
            self.answer_concept_recognizer = AnswerConceptRecognizerRNNBased.load(config.ANSWER_CONCEPT_RECOGNIZER_MODEL)
        return self.answer_concept_recognizer