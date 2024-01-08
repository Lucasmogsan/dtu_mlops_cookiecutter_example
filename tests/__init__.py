import os

_TEST_ROOT = os.path.dirname(os.path.realpath(__file__))  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of data
_NON_EXISTING_PATH = os.path.join(_PROJECT_ROOT, "non_existing_path")
