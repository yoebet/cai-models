import os
import re
from datetime import datetime

import pandas as pd
import numpy as np
import torchdata.datapipes as dp
from torch.utils.data import IterDataPipe
from copy import deepcopy
from tm.dataset.dataset_config import DatasetConfig, intervals_meta, extra_intervals_map


def interval_to_pandas_freq(interval):
    freq = interval.upper()
    if freq[-1] == 'M':
        freq = interval[0:-1] + 'T'
    return freq


dt_fields = ['month', 'date_day', 'week_day', 'time_hour']

v_price_fields = ['open', 'high', 'low', 'close']

v_fields = v_price_fields + [
    'volume', 'amount', 'trades',
    'buy_ratio', 'ch_percent', 'amp_percent'
]

target_fields = ['b_ts', 'b_price', 'f1_close']


@dp.functional_datapipe("load_kline_seq")
class KlineSeqIterDataPipe(IterDataPipe):

    def __init__(self, datapipe, ds_config: DatasetConfig, split_name):
        self.datapipe = datapipe
        self.ds_config = ds_config
        self.seq_len = ds_config.seq_len
        self.interval = ds_config.interval
        self.base_dir = ds_config.base_dir
        self.extra_intervals = ds_config.extra_intervals
        self.freq = interval_to_pandas_freq(self.interval)
        self.split_name = split_name
        self.file_no = 0

    def __iter__(self):

        interval_all_in_one_df_map = {}
        ds_name = self.split_name if self.split_name is not None else 'all'

        for filename in self.datapipe:
            dirname = os.path.dirname(filename)
            base_filename = os.path.basename(filename)
            print(f'{ds_name} {self.file_no}, {self.interval}, {base_filename}')

            self.file_no += 1
            df = pd.read_csv(filename, index_col='time')

            rolling_interval = None
            pi = '[1-9][0-9]?[smhd]'
            # BTCUSDT-1d-2018-02.csv
            r = rf'(\w+)-({pi})-(20\d\d)-(\d\d).csv'
            m = re.match(r, base_filename)
            if m:
                m_symbol, m_interval, m_year, m_month = m.groups()
                n_year, n_month = int(m_year), int(m_month)
                if n_month > 1:
                    n_month -= 1
                else:
                    n_month = 12
                    n_year -= 1
                s_month = str(n_month).rjust(2, '0')
                previous_filename = f'{dirname}/{m_symbol}-{m_interval}-{n_year}-{s_month}.csv'
                if os.path.exists(previous_filename):
                    df_previous = pd.read_csv(previous_filename, index_col='time')
                    df_previous = df_previous.iloc[1 - self.seq_len:]
                    df = pd.concat([df_previous, df])
            else:
                # BTCUSDT-1d-rolling-1h-m5.csv
                r = rf'(\w+)-({pi})-rolling-({pi})-m(\d).csv'
                m = re.match(r, base_filename)
                if m:
                    m_symbol, m_interval, r_interval, r_m = m.groups()
                    rolling_interval = r_interval

            extra_interval_files = []
            extra_interval_df_map = {}

            if self.extra_intervals is not None:
                for extra_interval in self.extra_intervals:
                    if interval_all_in_one_df_map.get(extra_interval) is not None:
                        pn = None
                    else:
                        im = intervals_meta.get(extra_interval)
                        if im is not None and im.get('one_file'):
                            symbol = self.ds_config.symbol
                            pn = f'{self.base_dir}/{extra_interval}-all/{symbol}/{symbol}-{extra_interval}-all.csv'
                        else:
                            pn = filename.replace(self.interval, extra_interval)
                        if not os.path.exists(pn):
                            p_bf = os.path.basename(pn)
                            print(f'{p_bf} not found')
                            break

                    extra_freq = interval_to_pandas_freq(extra_interval)
                    extra_interval_files.append((extra_interval, extra_freq, pn))

                if len(extra_interval_files) != len(self.extra_intervals):
                    break

            for (extra_interval, extra_freq, pn) in extra_interval_files:

                df2 = interval_all_in_one_df_map.get(extra_interval)

                if df2 is None:
                    df2 = pd.read_csv(pn, index_col='time')
                    df2.index = pd.PeriodIndex(data=df2.index, freq=extra_freq)
                    # temp
                    df2 = df2[~df2.index.duplicated()]
                    df2 = df2.ffill()
                    df2.index = df2.index.to_timestamp()

                    df2['ch_percent'] = (df2['close'] - df2['open']) / df2['open'] * 100
                    df2['amp_percent'] = (df2['high'] - df2['low']) / df2['low'] * 100
                    df2['buy_ratio'] = df2['buy_vol'] / df2['volume']

                    im = intervals_meta.get(extra_interval)
                    if im is not None and im.get('one_file'):
                        interval_all_in_one_df_map[extra_interval] = df2
                    else:
                        # load previous
                        ...

                extra_interval_df_map[extra_interval] = df2

            # in case rolling
            df['b_time'] = pd.to_datetime(df.index)
            df.index = pd.PeriodIndex(data=df.index, freq=self.freq)
            if rolling_interval is None:
                # temp
                df = df[~df.index.duplicated()]
                df = df.ffill()
            b_time = df['b_time']

            df['month'] = df.index.month / 12
            df['date_day'] = (df.index.day + 1) / 31
            df['week_day'] = (df.index.weekday + 1) / 7
            df['time_hour'] = (df.index.hour + 1) / 24

            df['ch_percent'] = (df['close'] - df['open']) / df['open'] * 100
            df['amp_percent'] = (df['high'] - df['low']) / df['low'] * 100
            df['buy_ratio'] = df['buy_vol'] / df['volume']

            df = df[dt_fields + v_fields]

            # target
            df['b_ts'] = b_time.map(pd.Timestamp.timestamp)
            df['b_price'] = df['open']
            f1_close = df['close']
            f1_close.index = f1_close.index.shift(-1)
            f1_close = f1_close.iloc[1:]
            df = df.iloc[:-1]
            b_time = b_time[:-1]
            df['f1_close'] = f1_close
            target_fields_count = len(target_fields)
            i_f1_close = -1

            fi = lambda n: df.columns.get_loc(n)
            i_open = fi('open')
            i_volume = fi('volume')
            i_amount = fi('amount')
            i_trades = fi('trades')
            i_high = fi('high')
            i_low = fi('low')
            i_close = fi('close')

            seq_len = self.seq_len
            nv = df.values
            for i in range(nv.shape[0] - seq_len):
                # print(i)
                seq = nv[i:i + seq_len]
                if np.isnan(np.sum(seq)):
                    continue
                seq = deepcopy(seq)

                time_to = b_time[i + seq_len - 1]
                not_completed = False

                for (extra_interval, extra_freq, _pn) in extra_interval_files:
                    df2 = extra_interval_df_map[extra_interval]
                    df2_to_idx = df2.index.searchsorted(time_to, side='right')
                    if df2_to_idx < seq_len:
                        not_completed = True
                        break
                    df2 = df2[df2_to_idx - seq_len:df2_to_idx]
                    df2 = df2[v_fields]
                    nv2 = deepcopy(df2.values)
                    seq = np.insert(seq, -target_fields_count, nv2.T, axis=1)

                if not_completed:
                    continue

                # encode seq
                bp = seq[-1, i_open]
                b_volume = np.mean(seq[:, i_volume])
                b_amount = np.mean(seq[:, i_amount])
                b_trades = np.mean(seq[:, i_trades])
                vl = len(v_fields)
                for ii in range(len(extra_interval_files) + 1):
                    fii = vl * ii
                    seq[:, i_open + fii] /= bp
                    seq[:, i_high + fii] /= bp
                    seq[:, i_low + fii] /= bp
                    seq[:, i_close + fii] /= bp
                    seq[:, i_volume + fii] /= b_volume
                    seq[:, i_amount + fii] /= b_amount
                    seq[:, i_trades + fii] /= b_trades
                seq[:, i_f1_close] /= bp

                if not np.isnan(np.sum(seq)):
                    yield seq


def to_time_str(d):
    dt = datetime.utcfromtimestamp(round(d))
    return dt.isoformat(sep=' ', timespec='minutes')


def decode_seq(seq, interval):
    columns = dt_fields + v_fields
    ext_intervals = extra_intervals_map[interval]
    for ei in ext_intervals:
        efs = [f'{ei}_{f}' for f in v_fields]
        columns += efs
    columns += target_fields
    df = pd.DataFrame(seq, columns=columns)
    df['month'] *= 12
    df['date_day'] = df['date_day'] * 31 - 1
    df['week_day'] = df['week_day'] * 7 - 1
    df['time_hour'] = df['time_hour'] * 24 - 1
    df['b_ts'] = df['b_ts'].apply(to_time_str)

    brow = df.iloc[-1]
    bp = brow['b_price']
    df['f1_close'] *= bp
    for pf in v_price_fields:
        df[pf] *= bp
        for ei in ext_intervals:
            df[f'{ei}_{pf}'] *= bp

    return df
