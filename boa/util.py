import datetime
import sys


def print_message(msg: str):
    date_format = '%Y-%m-%d %H:%M:%S'
    date = datetime.datetime.now().strftime(date_format)
    print('{date}  {msg}'.format(date=date, msg=msg))
    sys.stdout.flush()
