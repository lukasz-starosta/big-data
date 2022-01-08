def get_race_number(race):
    race_dictionary = {
        "AMERICAN INDIAN/ALASKAN NATIVE": 0,
        "OTHER": 0,
        "ASIAN / PACIFIC ISLANDER": 1,
        "BLACK": 2,
        "BLACK HISPANIC": 3,
        "WHITE HISPANIC": 4,
        "WHITE": 5,
    }
    return race_dictionary[race]


def get_sex_number(sex):
    sex_dictionary = {
        "F": 0,
        "E": 1,
        "D": 2,
        "M": 3,
    }
    return sex_dictionary[sex]