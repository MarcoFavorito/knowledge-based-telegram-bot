from exceptions import ConceptNotFound
from kbs.DBManager import DBManager
from kbs.models import Pattern, Relation
from messages.Messages import Messages
from models.ModelManager import ModelManager
from models.answer_generation.AnswerGenerator import AnswerGenerator
from utils.workflow.ConceptSelector import ConceptSelector
from workflow.State import State
from workflow.Job import Job
import config
import utils.misc as misc
import constants as c
import re

class QueryAnswerGenerationJob(Job):
    """
    1) Concept retrieval
    2) Answer Generation
    3) Question pattern retrieval

    """
    def validate_input(self, input_text) -> bool:
        return True

    def the_job(self, input_text):
        model_manager = ModelManager()
        context = self.workflow_manager.getContext()

        query = context.query
        relation = context.relation


        type2concepts = context.type2concepts
        c1, c2 = misc.extract_concepts(type2concepts, from_question=True)
        self.joblog("Extracted concepts are: c1={0}, c2={1}".format(repr(c1), repr(c2)))

        if not c1 and not c2:
            self.joblog("Extraction failed. No valid concept has been found. I'm terribly sorry.")
            self.workflow_manager.send_message(Messages.CONCEPT_RECOGNITION_FAILED.value)
            return

        # ------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------
        # Concept retrieval
        # ------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------

        cs = ConceptSelector()

        if c1 and c2:
            #Binary question
            is_found = cs.retrieve_binary_answer(c1, c2, relation)
            ans = "yes" if is_found else "no"
            self.workflow_manager.send_message(ans)
            self.joblog("Binary answer is: %s" %repr(ans))
            return
        else:
            #Non-binary question (don't say)
            try:
                c2 = cs.get_right_concept(c1, relation)
            except ConceptNotFound:
                self.joblog("Right concept not found. :(")
                self.workflow_manager.send_message(Messages.Q_CONCEPT_RETRIEVAL_FAILED.value)
                return

        self.joblog("Retrieved concepts are: c1={0}, c2={1}".format(repr(c1),repr(c2)))

        # ------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------
        # Answer generation
        # ------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------

        # Query preprocessing: make it in the form
        transformed_query = query
        if c1:
            transformed_query = re.sub(c.CONCEPT_PATTERN.format(c1), c.LEFT_CONCEPT_TAG, transformed_query)

        if c2:
            transformed_query = re.sub(c.CONCEPT_PATTERN.format(c2), c.RIGHT_CONCEPT_TAG, transformed_query)

        if transformed_query[-1]=="?":
            transformed_query = transformed_query[:-1] +" ?"
        self.joblog("Transformed query: %s" % transformed_query)

        ag = model_manager.get_answer_generator()

        generated_answer = ag.evaluate(transformed_query)
        self.joblog("Generated answer: %s" % str(generated_answer[0]))

        generated_answer = " ".join(generated_answer[0][:-1])
        rectified_answer = misc.rectify_answer(generated_answer)
        self.joblog("Rectified answer: %s" %  rectified_answer)

        transformed_answer = rectified_answer
        transformed_answer = re.sub(c.CONCEPT_PATTERN.format(c.LEFT_CONCEPT_TAG), c1, transformed_answer)
        transformed_answer = re.sub(c.CONCEPT_PATTERN.format(c.RIGHT_CONCEPT_TAG), c2, transformed_answer)
        self.joblog("Transformed answer: %s" %  transformed_answer)

        self.workflow_manager.send_message(transformed_answer)

        # ------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------
        # Question Pattern retrieval: store a new question pattern
        # ------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------

        # context.query here is the received query
        new_question_pattern = transformed_query
        new_question_pattern = new_question_pattern.replace(c.LEFT_CONCEPT_TAG, "X")
        new_question_pattern = new_question_pattern.replace(c.RIGHT_CONCEPT_TAG, "Y")


        self.joblog("Detected new question pattern: %s of relation %s" %(new_question_pattern, relation))
        self.joblog("Checking if it is already present.")
        db = DBManager()

        chosen_relation_obj = db.session.query(Relation).filter(Relation.simple_name==relation).one()

        patterns = db.session.query(Pattern).filter(Pattern.relation==chosen_relation_obj)\
            .filter(Pattern.question.like(new_question_pattern)).all()
        if len(patterns)!=0:
            self.joblog("Pattern already present. Skip insertion.")
        else:
            new_pattern = Pattern(question=new_question_pattern, relation=chosen_relation_obj)
            db.session.add(new_pattern)
            db.session.commit()
            self.joblog("Pattern added: %s" %new_question_pattern)

        return




    def get_new_state(self):
        return self.workflow_manager._dispatch(State.FINISH_WORKFLOW, "")