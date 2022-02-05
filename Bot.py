# -------------------- Imports --------------------
import os
from dotenv import load_dotenv
load_dotenv()

import logging
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import predict

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

logger = logging.getLogger(__name__)


def start(update : Update, context:CallbackContext) -> None:
    update.message.reply_text('Hi send an image to classify!')

def help(update: Update, context: CallbackContext) -> None:
    update.message.reply_text("I'm pictures recognition Bot. I can detect about 5 classes of objects! Just send me a random photo :)")


def photo(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    photoFile = update.message.photo[-1].get_file()
    photoFile.download("UserPhoto.jpg")
    logger.info("Photo of %s: %s", user.first_name, 'UserPhoto.jpg')
    update.message.reply_text(
        'Okay now wait a few seconds!!!'
    )

    DC = predict.DiseaseClassification(filename = "UserPhoto.jpg")
    result, classLabel = DC.predictionDisease()
    ClassName = DC.className(classLabel)

    update.message.reply_text("With " + str(max(result[0][classLabel])) + "% " + "accuracy, this is a " + ClassName)


def main():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    TOKEN = os.getenv("API_KEY")
    updater = Updater(TOKEN, use_context=True)
    PORT = int(os.environ.get('PORT', '5000'))

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help))

    # on noncommand i.e message - echo the message on Telegram
    dispatcher.add_handler(MessageHandler(Filters.photo & ~Filters.command, photo))

    updater.start_webhook(listen="0.0.0.0",
                      port=PORT,
                      url_path=TOKEN)
    updater.bot.set_webhook("https://gentle-coast-64239.herokuapp.com/" + TOKEN)
    updater.idle()


if __name__ == "__main__":
    main()
