from abc import ABC,ABCMeta, abstractmethod
# from workflow.WorkflowManager import WorkflowManager
from exceptions import UnknownStateException
from workflow.State import State
import utils.mylogger as mylogger

class Job(ABC):
    def __init__(self, workflow_manager) -> None:
        self.workflow_manager = workflow_manager


    def __call__(self, input_text):
        """
        Called into WorkflowManager.
        :param input_text:
        :return: the new state for the WorkflowManager
        """
        self.preprocess()
        if not self.validate_input(input_text):
            self.joblog("Input is not valid: %s" % repr(input_text))
            return self.error_callback(input_text)
        else:
            # self.joblog("Input is valid for the current Job: %s" % repr(input_text))
            # self.joblog("Entering 'the_job()'...")
            self.the_job(input_text)
            # self.joblog("Saving context...")
            # self.save_context(input_text)
            self.postprocess()
            # self.joblog("Job done!")
            return self.get_new_state()


    @abstractmethod
    def the_job(self, *args, **kwargs) -> None:
        raise Exception("Abstract method not implemented")

    @abstractmethod
    def get_new_state(self) -> State:
        raise Exception("Abstract method not implemented")

    # def save_context(self, input_text):
    #     pass

    def validate_input(self, input_text) -> bool:
        return True

    def error_callback(self, input_text) -> State:
        raise UnknownStateException()

    def preprocess(self):
        self.joblog("entering...")
        pass

    def postprocess(self):
        self.joblog("exiting...")
        pass

    def joblog(self, msg, lvl:int=mylogger.INFO):
        self.workflow_manager.logger.log(lvl, str(self.__class__.__name__) + ": " + msg)
