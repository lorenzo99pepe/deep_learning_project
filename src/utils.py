from pathlib import Path
import os

TYPE_NAMES = ['t2', 't1', 't1ce', 'flair', 'seg']

LOGS_FILE_PATH = Path(os.getcwd()) / 'models' / 'log.csv'