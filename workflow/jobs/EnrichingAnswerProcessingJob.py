from exceptions import UnknownStateException
from kbs.ApiManager import ApiManager
from kbs.models import Item
from messages.TelegramMessagesHelper import TelegramMessagesHelper
from models.ModelManager import ModelManager
from models.concept_recognizer.ConceptRecognizerRNNBased import ConceptRecognizerRNNBased
from workflow.State import State
from workflow.Job import Job
from messages.Messages import Messages
from messages.MessageFactory import MessageFactory
import numpy as np
import config
import utils.misc as misc



class EnrichingAnswerProcessingJob(Job):

    def the_job(self, input_text):
        model_manager = ModelManager()
        context = self.workflow_manager.getContext()

        answer = input_text
        relation = context.relation

        #Do the processing, i.e. send to server the update with the user answer

        # Concept recognition
        answer_concept_recognizer = model_manager.get_answer_concept_recognizer()
        type2concepts = answer_concept_recognizer.predict(answer, relation)
        self.joblog("Predicted concepts are: " + str(type2concepts))


        c1, c2 = misc.extract_concepts(type2concepts, from_question=False)
        self.joblog("The recognized concepts are: left=%s, right=%s" % (c1, c2))

        if not c2:
            self.joblog("Extraction failed. No right concept has been found.")
            self.workflow_manager.send_message(Messages.CONCEPT_RECOGNITION_FAILED.value)
            return

        # context.c1 = c1
        context.c2 = c2

        # Update the KB
        query = context.query
        domain = context.domain
        c1 = context.c1

        new_item = Item(
                question = query,
                domains=domain,
                relation=relation,
                answer=answer,
                c1=c1,
                c2=c2,
                context=""
            )

        self.joblog("Sending the update to KB: %s" % new_item.to_str())
        api_man = ApiManager.init_from_conf(config.API_CONF)
        response = api_man.add_item(new_item, is_test=False)
        self.joblog("Response: %s" % str(response))
        context.last_id_inserted = response



    def get_new_state(self):
        return self.workflow_manager._dispatch(State.FINISH_WORKFLOW, "")

