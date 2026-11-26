# main.py

import multiprocessing


def main():
    from gui import run_app
    run_app()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
