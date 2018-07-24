from telepot.namedtuple import KeyboardButton, ReplyKeyboardMarkup
from messages.Messages import Messages


class TelegramMessagesHelper(object):

    @staticmethod
    def get_reply_keyboard_markup(button_strings, resize_keyboard=True):
        keyboard_buttons = []
        for s in button_strings:
            keyboard_buttons+=[KeyboardButton(text=s)]

        #3*N
        keyboard_buttons_reshaped = []
        for i in range(0, len(keyboard_buttons), 3):
            keyboard_buttons_reshaped.append(keyboard_buttons[i:i+3])

        keyboard = ReplyKeyboardMarkup(keyboard=keyboard_buttons_reshaped, resize_keyboard=resize_keyboard,
                                       one_time_keyboard=True)
        return keyboard
