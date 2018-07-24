from exceptions import UnknownStateException
from kbs.DBManager import DBManager, Relation, Domain
from messages.TelegramMessagesHelper import TelegramMessagesHelper
from workflow.State import State
from workflow.Job import Job
from messages.Messages import Messages
# from workflow.WorkflowManager import WorkflowManager
import logging


class AskQueryEnrichSwitchJob(Job):
    """
    Ask to the user what he wants to do.
    """

    def the_job(self, input_text):
        """simply send the keyboard for the switching"""
        keyboard = TelegramMessagesHelper.get_reply_keyboard_markup([
                Messages.QUERYING_MODE_STRING.value, Messages.ENRICHING_MODE_STRING.value
            ])
        self.workflow_manager.send_message(Messages.QE_SWITCH_MSG.value, reply_markup=keyboard)



    def get_new_state(self):
        return State.QE_SWITCH

