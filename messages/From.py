class From(object):
    def __init__(self, id, first_name, last_name, is_bot, language_code):
        self.id = id
        self.first_name = first_name
        self.last_name = last_name
        self.is_bot = is_bot,
        self.language_code = language_code

    def __str__(self) -> str:
        return super().__str__()

    def __repr__(self) -> str:
        return super().__repr__()

    @classmethod
    def from_telegram_msg(cls, telegram_msg):
        from_ = telegram_msg["from"]

        id = from_["id"]
        first_name = from_["first_name"]
        last_name = from_["last_name"]
        is_bot = from_["is_bot"]
        language_code = from_["language_code"]

        return cls(id, first_name, last_name, is_bot, language_code)


