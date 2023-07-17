import os
import re

DS_SPLIT_METHOD_VAL_3M9M = 'val3m9m'
DS_SPLIT_METHOD_VAL_2023 = 'val2023'

DEFAULT_SPLIT_METHOD = DS_SPLIT_METHOD_VAL_2023


def filter_file_for_split(method_name, split_name='train'):
    def fp(filename):
        base_name = os.path.basename(filename)
        # BTCUSDT-1d-2022-03.csv
        fr = r'(\w+)-([1-9][0-9]?[smhd])-(20\d\d)-(\d\d).csv'
        m = re.match(fr, base_name)
        if m is None:
            return False
        _symbol, _interval, m_year, m_month = m.groups()
        year, month = int(m_year), int(m_month)
        if year == 2017:
            return False

        if method_name == DS_SPLIT_METHOD_VAL_2023:
            if split_name == 'train':
                return year <= 2022
            else:
                return (year * 100 + month) > 202301
        elif method_name == DS_SPLIT_METHOD_VAL_3M9M:
            if split_name == 'train':
                return month != 3 and month != 9
            else:
                return month == 3 or month == 9

        return False

    return fp


def filter_file_by_pattern(pat):
    def fp(filename):
        base_name = os.path.basename(filename)
        # return '-2022-' in fn
        if '-2017-' in base_name:
            return False
        return re.search(pat, base_name) is not None

    return fp


def filter_file_by_ym(i_years=None, i_months=None):
    def fp(filename):
        base_name = os.path.basename(filename)
        r = rf'.*-(20\d\d)-(\d\d).csv$'
        m = re.match(r, base_name)
        if m is None:
            return False
        m_year, m_month = m.groups()
        year, month = int(m_year), int(m_month)
        if i_years is not None and year not in i_years:
            return False
        if i_months is not None and month not in i_months:
            return False
        return True

    return fp
