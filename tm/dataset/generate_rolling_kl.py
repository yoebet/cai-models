import os
from datetime import datetime
import pandas as pd
from tm.dataset.dataset_config import SPLIT_NAME_TRAIN, SPLIT_NAME_VALIDATION
from tm.dataset.dataset_split import DS_SPLIT_METHOD_VAL_2023, DS_SPLIT_METHOD_VAL_3M9M
from tm.dataset.dataset_config import intervals_meta


def rolling_to_interval(base_dir,
                        symbol,
                        from_interval,
                        from_df,
                        rolling_to,
                        ds_split_name=None,
                        ds_split_method_name=None,
                        ):
    to_interval, times = rolling_to
    print('>', to_interval, ds_split_name)

    df2 = from_df.rolling(times, min_periods=times).agg({'time': lambda s: s.iloc[-1],
                                                         'open_time': lambda s: s.iloc[0],
                                                         'open_hour': lambda s: s.iloc[0],
                                                         'open': lambda s: s.iloc[0],
                                                         'high': 'max',
                                                         'low': 'min',
                                                         'close': lambda s: s.iloc[-1],
                                                         'volume': 'sum',
                                                         'amount': 'sum',
                                                         'buy_vol': 'sum',
                                                         'buy_amt': 'sum',
                                                         'trades': 'sum',
                                                         })
    df2 = df2.iloc[(times - 1):]
    i_span = intervals_meta[to_interval]['seconds']
    spans = df2['time'] - df2['open_time']
    span_neq = spans == i_span
    df2 = df2[span_neq]
    print('wi', (~span_neq).sum())

    def to_time_str(d):
        ts = round(d)
        dt = datetime.utcfromtimestamp(ts)
        return dt.isoformat(timespec='seconds')

    df2['time'] = df2['time'].apply(to_time_str)
    df2['open_time'] = df2['open_time'].apply(to_time_str)

    print(df2.head())

    base_filename = f'{symbol}-{to_interval}-rolling-{from_interval}'
    if ds_split_method_name is not None:
        split_target_dir = f'{base_dir}/{to_interval}-rolling-{from_interval}-{ds_split_method_name}-{ds_split_name}/{symbol}'
    else:
        split_target_dir = f'{base_dir}/{to_interval}-rolling-{from_interval}/{symbol}'

    os.makedirs(split_target_dir, exist_ok=True)

    # rows = df2.shape[0]
    for mod in range(times):
        # split_df = df2.iloc[range(mod, rows, times)]
        split_df = df2[df2['open_hour'] % times == mod]
        split_df = split_df.drop(['open_hour'], axis=1)
        split_df.to_csv(f'{split_target_dir}/{base_filename}-m{mod}.csv',
                        index=False,
                        )

    target_dir = f'{base_dir}/{to_interval}-all/{symbol}'
    if ds_split_method_name is not None:
        target_filename = f'{base_filename}-{ds_split_method_name}-{ds_split_name}.csv'
    else:
        target_filename = f'{base_filename}-all.csv'
    os.makedirs(target_dir, exist_ok=True)
    df2 = df2.drop(['open_hour'], axis=1)
    df2.to_csv(f'{target_dir}/{target_filename}',
               index=False,
               )


def build_rolling_kl(base_dir,
                     symbol,
                     from_interval,
                     to_intervals,
                     ds_split_name=None,
                     ds_split_method_name=None,
                     ):
    from_freq = from_interval.upper()

    if ds_split_name is None:
        src_filename = f'{symbol}-{from_interval}-all.csv'
    else:
        src_filename = f'{symbol}-{from_interval}-{ds_split_method_name}-{ds_split_name}.csv'
    df = pd.read_csv(f'{base_dir}/{from_interval}-all/{symbol}/{src_filename}',
                     # nrows=200,
                     index_col='time',
                     )
    print(df.head())
    print('total', df.shape[0], 'rows')
    print()

    open_time_period = pd.PeriodIndex(data=df['open_time'], freq=from_freq)
    df['open_hour'] = open_time_period.hour
    df['open_time'] = open_time_period.to_timestamp().map(pd.Timestamp.timestamp)
    df.index = pd.PeriodIndex(data=df.index, freq=from_freq)
    df = df[~df.index.duplicated()]
    df = df.ffill()
    time2 = df.index.to_timestamp().map(pd.Timestamp.timestamp)
    df.reset_index(inplace=True)
    df['time'] = time2

    for rolling_to in to_intervals:
        rolling_to_interval(base_dir,
                            symbol,
                            from_interval,
                            df,
                            rolling_to,
                            ds_split_name=ds_split_name,
                            ds_split_method_name=ds_split_method_name,
                            )
        print()


if __name__ == '__main__':
    bd = '../../data/tm/market/basic/spot-kline'
    split_method = DS_SPLIT_METHOD_VAL_2023
    split_names = [SPLIT_NAME_TRAIN, SPLIT_NAME_VALIDATION]
    for sym in ['BTCUSDT', 'ETHUSDT']:
        for split_name in split_names:
            build_rolling_kl(base_dir=bd,
                             symbol=sym,
                             from_interval='1h',
                             to_intervals=[('4h', 4), ('1d', 24)],
                             ds_split_name=split_name,
                             ds_split_method_name=split_method,
                             )
