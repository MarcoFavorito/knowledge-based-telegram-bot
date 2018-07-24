from messages.Chat import Chat
from messages.From import From


class Message(object):
    def __init__(self, id, date, text, Chat, From):
        self.id = id
        self.date = date
        self.text = text
        self.Chat = Chat
        self.From = From

    def __str__(self) -> str:
        return super().__str__()

    def __repr__(self) -> str:
        return super().__repr__()
