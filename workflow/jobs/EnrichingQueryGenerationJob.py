from exceptions import UnknownStateException
from messages.TelegramMessagesHelper import TelegramMessagesHelper
from utils.workflow.QueryGeneration import QueryGeneration
from workflow.State import State
from workflow.Job import Job
from messages.Messages import Messages
from messages.MessageFactory import MessageFactory


class EnrichingQueryGenerationJob(Job):
    def the_job(self, input_text):

        qg = QueryGeneration()

        chosen_domain = self.workflow_manager.getContext().domain
        self.joblog("The chosen domain is: %s" %chosen_domain)

        chosen_concept = qg.get_concept_by_domain(chosen_domain)
        self.joblog("The chosen concept is: %s" % chosen_concept)

        chosen_relation = qg.get_relation_by_concept(chosen_concept, chosen_domain)
        self.joblog("The chosen relation is: %s" % chosen_relation)

        query = qg.generate_query(chosen_concept, chosen_relation)
        self.joblog("The generated query is: %s" % query)

        self.workflow_manager.send_message(query)

        #Save into context
        context = self.workflow_manager.getContext()
        context.query = query
        context.domain = chosen_domain
        context.relation = chosen_relation
        context.c1 = chosen_concept

    def get_new_state(self):
        return State.E_ANSWER_PROCESSING

