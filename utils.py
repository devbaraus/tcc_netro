def merge_dicts(base_dict, *dicts):
    """
    Merge two dictionaries.
    """
    dict3 = base_dict.copy()

    for d in dicts:
        for key, value in d.items():
            dict3[key] = [*dict3[key], *value]

    return dict3


if __name__ == '__main__':
    base = {
        'a': [],
        'b': [],
        'c': [],
    }

    dic1 = {
        'a': [1, 2, 3],
        'b': [1, 2, 3],
        'c': [1, 2, 3],
    }

    dic2 = {
        'a': [4, 5, 6],
        'b': [4, 5, 6],
        'c': [4, 5, 6],
    }

    final_dict = merge_dicts(base, *[dic1, dic2])

    print(final_dict)
