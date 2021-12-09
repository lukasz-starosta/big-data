import pandas as pd

def get_rand_records(data, number):
    new_records = data.sample(n = number)
    return new_records