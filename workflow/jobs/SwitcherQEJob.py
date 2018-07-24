from exceptions import UnknownStateException
from messages.TelegramMessagesHelper import TelegramMessagesHelper
from workflow.State import State
from workflow.Job import Job
from messages.Messages import Messages
from messages.MessageFactory import MessageFactory


class SwitcherQEJob(Job):
    def validate_input(self, input_text) -> bool:
        """input is valid only if contains the text in the keyboard"""
        return input_text in [
            Messages.QUERYING_MODE_STRING.value,
            Messages.ENRICHING_MODE_STRING.value
        ]

    def the_job(self, input_text):
        """ackowledge the choice to the user and then:
        wait for the question of the user (if Querying mode)
        schedule the EnrichingQueryGenerationJob for generate a question to the user (if Enriching mode)"""
        self.joblog("The user want to ask a question?: %s" % repr(input_text))
        if input_text==Messages.QUERYING_MODE_STRING.value:
            self.workflow_manager.send_message(Messages.Q_WELCOME_MSG.value)
            self.new_state = State.Q_PROCESSING
        elif input_text==Messages.ENRICHING_MODE_STRING.value:
            self.workflow_manager.send_message(Messages.E_WELCOME_MSG.value)
            self.new_state = State.E_QUERY_GENERATION
        else:
            raise UnknownStateException()

    def get_new_state(self):
        # see the explanation above
        if self.new_state==State.Q_PROCESSING:
            return self.new_state
        elif self.new_state==State.E_QUERY_GENERATION:
            return self.workflow_manager._dispatch(self.new_state, "")
        else:
            raise UnknownStateException()



