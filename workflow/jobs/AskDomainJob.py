from exceptions import UnknownStateException
from kbs.DBManager import DBManager
from kbs.models import Domain
from messages.TelegramMessagesHelper import TelegramMessagesHelper
from workflow.State import State
from workflow.Job import Job
from messages.Messages import Messages


class AskDomainJob(Job):
    def the_job(self, input_text):
        """
        The job for ask the Domain to the user.
        """

        db = DBManager()
        self.joblog("Syncing with KB...")

        # If there is some previous inserted item,
        # update the local db
        last_id_inserted = self.workflow_manager.context.last_id_inserted
        if not last_id_inserted or last_id_inserted==-1:
            last_id_inserted=-1
        new_items = db.sync(from_id=last_id_inserted)

        if len(new_items)>0:
            self.joblog("The new item found are:")
            for i in new_items:
                self.joblog(i.to_str())
        else:
            self.joblog("No new item found.")

        # retrieve dynamically the available domains
        domains = db.session.query(Domain).all()
        keyboard = TelegramMessagesHelper.get_reply_keyboard_markup([d.simple_name for d in domains])
        self.workflow_manager.send_message(Messages.ASK_DOMAIN_MSG.value, reply_markup=keyboard)


    def get_new_state(self):
        # "Wait" for the next message, "scheduling" the right Job.
        return State.RECEIVE_DOMAIN

