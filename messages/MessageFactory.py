from pprint import pprint

from messages.Message import Message
from messages.Chat import Chat
from messages.From import From

class MessageFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def getMessageFromTelegramMsg(telegram_msg):
        chat = Chat.from_telegram_msg(telegram_msg)
        from_ = From.from_telegram_msg(telegram_msg)
        message = Message(telegram_msg["message_id"], telegram_msg["date"], telegram_msg["text"], chat, from_)
        return message

