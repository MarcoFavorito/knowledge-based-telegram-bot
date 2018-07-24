from telepot import Bot

from workflow.Context import Context
from workflow.State import State
from workflow.jobs import *
import utils.mylogger

class WorkflowManager(object):
    """
    This is the core class, which manages the workflow and the dispatching of telegram messages to the right Job.
    It is called by every Job for memorize data into the Context object and for sending messages to the user.
    """


    LOG_TAG = "WorkflowManager"

    @staticmethod
    def generate_state2job(workflow_manager):
        """
        Mapping between states and Jobs: it is used to dispatch the message
        to the correct Job, given the current state.
        :param workflow_manager:
        :return:
        """

        # Jobs classes are loaded dynamically (see jobs.__init__.py script)
        return {
            State.START: StartJob(workflow_manager),
            State.ASK_DOMAIN: AskDomainJob(workflow_manager),
            State.RECEIVE_DOMAIN: ReceiveDomainJob(workflow_manager),
            State.QE_ASK_SWITCH: AskQueryEnrichSwitchJob(workflow_manager),
            State.QE_SWITCH: SwitcherQEJob(workflow_manager),

            State.Q_PROCESSING : QueryProcessingJob(workflow_manager),
            State.Q_ANSWER_GENERATION : QueryAnswerGenerationJob(workflow_manager),

            State.E_QUERY_GENERATION: EnrichingQueryGenerationJob(workflow_manager),
            State.E_ANSWER_PROCESSING: EnrichingAnswerProcessingJob(workflow_manager),
            State.FINISH_WORKFLOW : FinishWorkflowJob(workflow_manager)

        }



    def __init__(self) -> None:
        # Current state. Update in process_message at every _dispatch
        self.state = State.START
        self.context = Context()
        self.state2job = self.generate_state2job(self)



    def process_message(self, message):
        """
        Dispatchses the job and wait for retrieve the new state:State in which the Workflow will be.
        :param message:
        """
        text = message.text
        new_state = self._dispatch(self.state, text)
        self.state = new_state


    def _dispatch(self, state, text):
        """

        :param state: the current state
        :param text: the message received, the input text
        :return: the new state returned by the executed job.
        """
        return self.state2job[state](text)

    def getContext(self) ->Context:
        return self.context

    def getState(self):
        return self.state


class TelegramWorkflowManager(WorkflowManager):
    """
    This class manages the interaction with Telegram
    """


    LOG_TAG = "TelegramWorkflowManager"

    def __init__(self, chat_id, bot:Bot) -> None:
        super().__init__()
        self.chat_id = chat_id
        self.bot = bot
        self.logger = utils.mylogger.getLoggerByChatId(chat_id)
        self.logger.info(self.LOG_TAG + ": started new Workflow with chat_id=%d" % chat_id)


    def process_message(self, message):
        """
        Dispatchses the job and wait for retrieve the new state:State in which the Workflow will be.
        :param message:
        """
        text = message.text
        new_state = self._dispatch(self.state, text)
        self.state = new_state


    def send_message(self, message, reply_markup=None):
        # Telepot API
        self.bot.sendMessage(self.chat_id, text=message, reply_markup=reply_markup)

class ConsoleWorkflowManager(WorkflowManager):
    """
    A workaround to run the project on console rather than on Telegram.
    """


    LOG_TAG = "ConsoleWorkflowManager"

    def __init__(self) -> None:
        super().__init__()
        self.chat_id = -1
        self.logger = utils.mylogger.getLoggerByChatId(self.chat_id)
        self.logger.info(self.LOG_TAG + ": started new Workflow on console.")


    def process_message(self, message):
        text = message
        new_state = self._dispatch(self.state, text)
        self.state = new_state


    def send_message(self, message, reply_markup=None):
        """Build the string from Telegram-like message (workaround)"""
        print(message)
        if reply_markup is None:
            return
        # reply markup on multiple rows
        i = 0
        for row in reply_markup.keyboard:
            for key in row:
                print(key.text)
                i+=1
