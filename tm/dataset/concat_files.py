import os
import pathlib
from tm.dataset.dataset_config import intervals_meta, SPLIT_NAME_TRAIN, SPLIT_NAME_VALIDATION
from tm.dataset.dataset_split import filter_file_for_split, DS_SPLIT_METHOD_VAL_2023, DS_SPLIT_METHOD_VAL_3M9M


def concat_interval_symbol(base_dir,
                           interval,
                           symbol,
                           headline=True,
                           ds_split_name=None,
                           ds_split_method_name=None,
                           ):
    src_dir = pathlib.Path(base_dir, interval, symbol)
    src_names = os.listdir(src_dir)
    src_names = filter(lambda n: n.endswith('.csv'), src_names)
    if ds_split_name is not None:
        file_filter = filter_file_for_split(ds_split_name, ds_split_method_name)
        src_names = filter(file_filter, src_names)
    src_names = sorted(src_names)

    tgt_dir = pathlib.Path(base_dir, interval + '-all', symbol)
    os.makedirs(tgt_dir, exist_ok=True)
    if ds_split_name is None:
        tgt_filename = f'{symbol}-{interval}-all.csv'
    else:
        tgt_filename = f'{symbol}-{interval}-{ds_split_method_name}-{ds_split_name}.csv'
    print(f'> {tgt_filename}')
    with open(tgt_dir / tgt_filename, 'w') as outfile:
        first_file = True
        for src_name in src_names:
            with open(src_dir / src_name) as infile:
                first_line = True
                for line in infile:
                    if headline and not first_file and first_line:
                        first_line = False
                        continue
                    outfile.write(line)
                    first_line = False
            first_file = False


def concat_all_symbols(base_dir,
                       interval,
                       headline=True,
                       ds_split_method_name=None,
                       ):
    src_interval_dir = pathlib.Path(base_dir, interval)
    symbols = os.listdir(src_interval_dir)
    for symbol in symbols:
        print(f'concat {interval}/{symbol} ...')
        if ds_split_method_name is None:
            split_names = [None]
        else:
            split_names = [SPLIT_NAME_TRAIN, SPLIT_NAME_VALIDATION]
        for split_name in split_names:
            concat_interval_symbol(base_dir,
                                   interval,
                                   symbol,
                                   headline=headline,
                                   ds_split_method_name=ds_split_method_name,
                                   ds_split_name=split_name,
                                   )


def concat_all_intervals_symbols(base_dir,
                                 headline=True,
                                 ds_split_method_name=None,
                                 ):
    for (interval, meta) in intervals_meta.items():
        if meta.get('one_file'):
            concat_all_symbols(base_dir, interval, headline=headline, ds_split_method_name=ds_split_method_name)


if __name__ == '__main__':
    market_base = '../../data/tm/market'
    kline_base = f'{market_base}/basic/spot-kline'
    split_methods = [DS_SPLIT_METHOD_VAL_2023, None]

    # concat_interval_symbol(kline_base, '1d', 'BTCUSDT',
    #                        ds_split_method_name=split_method,
    #                        ds_split_name=SPLIT_NAME_VALIDATION,
    #                        )
    # concat_all_symbols(kline_base, '15m',
    #                    ds_split_method_name=split_method,
    #                    )
    for split_method in split_methods:
        concat_all_intervals_symbols(kline_base,
                                     ds_split_method_name=split_method,
                                     )
