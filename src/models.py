from enum import Enum


class Algorithm(Enum):
    GB_AT = "GB-AT"
    MAGNITUDE_INVERTED_GB_AT = "M-inverted GB-AT"
    S_DIPS = "S-DIPS"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value
