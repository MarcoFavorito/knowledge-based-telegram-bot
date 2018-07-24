from kbs.DBManager import DBManager, Domain
from messages.Messages import Messages
from workflow.State import State
from workflow.Job import Job


class ReceiveDomainJob(Job):
    """
    Job for process the receied domain
    """

    def validate_input(self, input_text) -> bool:
        """ validate the input only if it corresponds to the simple name of a Domain """
        db = DBManager()
        domain_simple_names = list(map(lambda x: x.simple_name, db.session.query(Domain).all()))
        return input_text in domain_simple_names

    def the_job(self, input_text):
        # the input_text at this point should be the chosen domain
        chosen_domain_simple_name = input_text
        context = self.workflow_manager.getContext()
        context.domain = chosen_domain_simple_name
        self.joblog("The chosen domain is: %s" % repr(input_text))

    def get_new_state(self):
        # shedule the job AskQueryEnrichSwitchJob
        return self.workflow_manager._dispatch(State.QE_ASK_SWITCH, "")

    def error_callback(self, input_text) -> State:
        # If the input is not valid, say it to the user and ask again the domain
        self.workflow_manager.send_message(Messages.DOMAIN_NOT_VALID.value)
        return self.workflow_manager._dispatch(State.ASK_DOMAIN, "")


