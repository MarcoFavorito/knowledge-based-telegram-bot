import telepot
from telepot.loop import MessageLoop

import config as conf
from handlers import handler as hdl

# chatId = 188997439
def main():
    bot = telepot.Bot(conf.TOKEN)

    a = bot.getMe()
    print(a)

    botHandler = hdl.BotHandler()
    MessageLoop(
        bot,
        {
            'chat': botHandler.mainHandler
        }
    ).run_as_thread()

    # The main process will block here; the handler for new messages runs as thread (see previous instruction)
    while True:
        input()


if __name__ == '__main__':
    main()