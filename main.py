# main.py

from gui import run_app

def main():
    run_app()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()