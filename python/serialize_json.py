#!/usr/bin/env python

# Copyright (c) 2013, Christopher R. Wagner
#  
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#  
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#  
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""JSON serialization routines for python data types.

Retains type information for basic python datatypes that do not have a
direct mapping to JSON without abusing JSON too much. Format is
similar to the one used by jsonpickle, though not guaranteed to be
compatible.

Currently handles dictionaries with non-string-keys, tuples, sets,
numpy arrays, namedtuples, and OrderedDicts.
"""

from collections import Iterable, OrderedDict, namedtuple

import numpy as np
import simplejson as json

MyTuple = namedtuple("MyTuple", "foo baz")

TEST_DATA = [
    1,
    2,
    3,
    23.32987,
    478.292222,
    -0.0002384,
    "testing",
    False,
    [4, 5, 6, [7, 8], 9],
    ("mixed", 5, "tuple"),
    {
        "str": 1,
        "str2": 2
    },
    {
        1: "str",
        2: "str4",
        (5, 6): "str8"
    },
    {4, 8, 2, "string", (4, 8, 9)},
    None,
    MyTuple(
        foo=1, baz=2),
    OrderedDict([('my', 23), ('order', 55), ('stays', 44), ('fixed', 602)]),
    np.array([[1, 2, 3], [4, 5, 6]]),
    np.array([[1.2398, 2.4848, 3.484884], [4.10, 5.3, 6.999992]]),
]


def nested_equal(v1, v2):
    """Compares two complex data structures.

    This handles the case where numpy arrays are leaf nodes.
    """
    if isinstance(v1, basestring) or isinstance(v2, basestring):
        return v1 == v2
    if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
        return np.array_equal(v1, v2)
    if isinstance(v1, dict) and isinstance(v2, dict):
        return nested_equal(v1.items(), v2.items())
    if isinstance(v1, Iterable) and isinstance(v2, Iterable):
        return all(nested_equal(sub1, sub2) for sub1, sub2 in zip(v1, v2))
    return v1 == v2


def isnamedtuple(obj):
    """Heuristic check if an object is a namedtuple."""
    return isinstance(obj, tuple) \
           and hasattr(obj, '_fields') \
           and hasattr(obj, '_asdict') \
           and callable(obj._asdict)


def serialize(data):
    if data is None or isinstance(data, (bool, int, float, str)):
        return data
    if isinstance(data, list):
        return [serialize(val) for val in data]
    if isinstance(data, OrderedDict):
        return {
            "py/collections.OrderedDict":
            [[serialize(k), serialize(v)] for k, v in data.iteritems()]
        }
    if isnamedtuple(data):
        return {
            "py/collections.namedtuple": {
                "type": type(data).__name__,
                "fields": list(data._fields),
                "values": [serialize(getattr(data, f)) for f in data._fields]
            }
        }
    if isinstance(data, dict):
        if all(isinstance(k, basestring) for k in data):
            return {k: serialize(v) for k, v in data.iteritems()}
        return {
            "py/dict": [[serialize(k), serialize(v)]
                        for k, v in data.iteritems()]
        }
    if isinstance(data, tuple):
        return {"py/tuple": [serialize(val) for val in data]}
    if isinstance(data, set):
        return {"py/set": [serialize(val) for val in data]}
    if isinstance(data, np.ndarray):
        return {
            "py/numpy.ndarray": {
                "values": data.tolist(),
                "dtype": str(data.dtype)
            }
        }
    raise TypeError("Type %s not data-serializable" % type(data))


def restore(dct):
    if "py/dict" in dct:
        return dict(dct["py/dict"])
    if "py/tuple" in dct:
        return tuple(dct["py/tuple"])
    if "py/set" in dct:
        return set(dct["py/set"])
    if "py/collections.namedtuple" in dct:
        data = dct["py/collections.namedtuple"]
        return namedtuple(data["type"], data["fields"])(*data["values"])
    if "py/numpy.ndarray" in dct:
        data = dct["py/numpy.ndarray"]
        return np.array(data["values"], dtype=data["dtype"])
    if "py/collections.OrderedDict" in dct:
        return OrderedDict(dct["py/collections.OrderedDict"])
    return dct


def data_to_json(data):
    return json.dumps(serialize(data))


def json_to_data(s):
    return json.loads(s, object_hook=restore)


def test_equivalence():
    if not nested_equal(TEST_DATA, json_to_data(data_to_json(TEST_DATA))):
        for element in TEST_DATA:
            serialized_element = json_to_data(data_to_json(element))
            if element != serialized_element:
                print("Mismatch: %s != %s" % (element, serialized_element))
    else:
        print("Success.")
        print(data_to_json(TEST_DATA))
        print("\nhas unserialized to\n")
        print(json_to_data(data_to_json(TEST_DATA)))


if __name__ == "__main__":
    test_equivalence()
