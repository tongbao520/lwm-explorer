import re
import yaml


def replace_e_float(d):
    p = re.compile(r"^-?\d+(\.\