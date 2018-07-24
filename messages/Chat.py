class Chat(object):
    def __init__(self, id, first_name, last_name, type):
        self.id = id
        self.first_name = first_name
        self.last_name = last_name
        self.type = type

    def __str__(self) -> str:
        return super().__str__()

    def __repr__(self) -> str:
        return super().__repr__()

    @classmethod
    def from_telegram_msg(cls, telegram_msg):
        chat = telegram_msg["chat"]

        id = chat["id"]
        first_name = chat["first_name"]
        last_name = chat["last_name"]
        type = chat["type"]

        return cls(id, first_name, last_name, type)
