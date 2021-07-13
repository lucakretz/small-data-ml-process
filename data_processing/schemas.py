"""Fix schemas for the produced and supported annotation formats."""

from schema import Schema, And

COORDINATE_NAMES = ["xmin", "ymin", "xmax", "ymax"]

SCHEMAS = {
    "label": Schema([(str, int)]),
    "bndbox": Schema(
        And(
            list,
            lambda x: all(isinstance(t, tuple) for t in x),
            lambda y: all(isinstance(t[0], str) and isinstance(t[1], dict) for t in y),
            lambda z: all(
                sorted(list(t[1].keys())) == sorted(COORDINATE_NAMES) for t in z
            ),
            lambda i: all(isinstance(v, float) for t in i for v in t[1].values()),
        )
    ),
    "polygon": Schema(
        [
            (
                str,
                And(
                    list,
                    lambda l: len(l) > 4,
                    lambda l: all(isinstance(float, item) for item in l),
                ),
            )
        ]
    ),
    "mask": Schema(
        And(
            list,
            lambda x: all(isinstance(t, tuple) for t in x),
            lambda y: all(isinstance(t[0], str) and isinstance(t[1], list) for t in y),
            lambda i: all(
                isinstance(v[0], int) and isinstance(v[1], int) for t in i for v in t[1]
            ),
        )
    ),
}

ANNOTATION_TYPE = "type"
ANNOTATION_STR = "class"
ANNOTATION_ENCODING = "encoding"
