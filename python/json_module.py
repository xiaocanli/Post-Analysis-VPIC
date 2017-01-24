"""
Module to deal with json file
"""
import simplejson as json

from serialize_json import data_to_json, json_to_data


def read_data_from_json(fname):
    """Read jdote data from a json file

    Args:
        fname: file name of the json file of the jdote data.
    """
    with open(fname, 'r') as json_file:
        data = json_to_data(json.load(json_file))
    print("Reading %s" % fname)
    return data

if __name__ == "__main__":
    pass
