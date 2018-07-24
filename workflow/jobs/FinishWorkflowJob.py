from exceptions import UnknownStateException
from messages.TelegramMessagesHelper import TelegramMessagesHelper
from workflow.State import State
from workflow.Job import Job
from messages.Messages import Messages
from messages.MessageFactory import MessageFactory


class FinishWorkflowJob(Job):

    def the_job(self, input_text):
        #Acknowledge of end of workflow
        # self.workflow_manager.send_message("Done!")
        pass


    def get_new_state(self):
        #Loop
        return self.workflow_manager._dispatch(State.ASK_DOMAIN, "")

