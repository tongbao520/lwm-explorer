import re
import yaml


def replace_e_float(d):
    p = re.compile(r"^-?\d+(\.\d+)?e-?\d+$")
    for name, val in d.items():
        if type(val) == dict:
           