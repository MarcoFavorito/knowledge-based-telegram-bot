from workflow.State import State
from workflow.Job import Job
from messages.Messages import Messages



class StartJob(Job):
    """
    The first job to be executed
    it simply send a welcome message to the user
    and dispatch the next job: the one who asks the domain
    """
    def validate_input(self, input_text) -> bool:
        # the first message has to be "/start"
        return input_text in [
            Messages.START.value
        ]

    def the_job(self, input_text):
        self.workflow_manager.send_message(Messages.WELCOME.value)


    def get_new_state(self):
        #Instead of return the new state, dispatch again!
        # return self.workflow_manager._dispatch(State.QE_ASK_SWITCH, "")
        return self.workflow_manager._dispatch(State.ASK_DOMAIN, "")


