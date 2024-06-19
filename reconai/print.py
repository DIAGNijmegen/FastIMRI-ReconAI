from datetime import datetime

from reconai import version


start = None


def print_log(*messages: str):
    global start
    if not start:
        start = datetime.now()

    for msg in messages:
        dt = str(datetime.now() - start)
        print(f'{dt:<10} | {msg}')
    print()


def print_version(*args):
    print_log(f'reconai version {version}', *args)