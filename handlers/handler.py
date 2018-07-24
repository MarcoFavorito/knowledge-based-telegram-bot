import telepot
import config as conf
from pprint import pprint
from messages.MessageFactory import MessageFactory


from telepot.namedtuple import InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup, KeyboardButton
import messages.Messages as msg_type
from workflow.WorkflowManager import TelegramWorkflowManager
import utils.mylogger as mylogger


START_MSG = "/start"
HELP_MSG = "/help"
SETTINGS_MSG = "/settings"
bot = telepot.Bot(conf.TOKEN)

class BotHandler():

    def __init__(self):
        self.processes = {}

    def mainHandler(self, telegram_msg):
        """
        The entry point for every incoming message
        It simply dispatch the message to the correct chat
        Look at the process.process_message(message)

        :param telegram_msg: Telegram data structure
        :return:
        """

        # Utils for retrieve the message from the telegram data structure
        message = MessageFactory.getMessageFromTelegramMsg(telegram_msg)

        process_id = message.Chat.id
        mylogger.getLoggerByChatId(process_id).info("Message: " + str(telegram_msg))

        if (process_id not in self.processes):
            # Initialization of the main object which manages all the workflow
            process = TelegramWorkflowManager(process_id, bot)
            self.processes[process_id] = process

        process = self.processes[process_id]
        process.process_message(message)


class ConsoleHandler():
    # TODO implement console handler
    def main(self):
        pass