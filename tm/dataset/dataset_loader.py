import os
import re

import torchdata.datapipes as dp
from torchdata.datapipes.iter import IterableWrapper
from torch.utils.data import DataLoader

from tm.dataset.dataset_config import DatasetConfig, extra_intervals_map, intervals_meta, SPLIT_NAME_TRAIN, \
    SPLIT_NAME_VALIDATION
from tm.dataset.kl_seq_iter_dp import KlineSeqIterDataPipe, decode_seq
from tm.dataset.dataset_split import filter_file_for_split, DEFAULT_SPLIT_METHOD
from tm.common.utils import set_print_options, check_batch


def load_dataset(ds_config: DatasetConfig,
                 filename_filter=None,
                 ds_split_name=None,
                 iterate_one_file=False,
                 rolling=None):
    dsc = ds_config
    interval = dsc.interval
    split_method = dsc.ds_split_method_name
    if iterate_one_file and rolling is None:
        idir = f'{dsc.base_dir}/{interval}-all/{dsc.symbol}'
        fbase = f'{dsc.symbol}-{interval}'
        if split_method is None or ds_split_name is None:
            file = f'{idir}/{fbase}-all.csv'
        else:
            file = f'{idir}/{fbase}-{split_method}-{ds_split_name}.csv'
        datapipe = IterableWrapper([file])
    else:
        if rolling is None:
            idir = interval
        else:
            idir = f'{interval}-rolling-{rolling}-{split_method}-{ds_split_name}'
        dir = f'{dsc.base_dir}/{idir}/{dsc.symbol}'
        datapipe = dp.iter.FileLister(dir,
                                      non_deterministic=True,
                                      masks='*.csv',
                                      )
    if filename_filter is not None:
        datapipe = datapipe.filter(filter_fn=filename_filter)
    assert KlineSeqIterDataPipe
    datapipe = datapipe.load_kline_seq(dsc, ds_split_name)
    datapipe = datapipe.shuffle(buffer_size=dsc.shuffle_buffer)
    dl = DataLoader(dataset=datapipe,
                    batch_size=dsc.batch_size,
                    drop_last=True,
                    num_workers=0
                    )
    return dl


def load_datasets(ds_config: DatasetConfig):
    imeta = intervals_meta[ds_config.interval]
    rolling = imeta.get('rolling')
    iterate_one_file = imeta.get('iterate_one_file')
    split_method = ds_config.ds_split_method_name
    if split_method is None:
        split_method = DEFAULT_SPLIT_METHOD
        ds_config.ds_split_method_name = split_method
    if iterate_one_file:
        t_filename_filter = None
        v_filename_filter = None
    else:
        t_filename_filter = filter_file_for_split(split_method, SPLIT_NAME_TRAIN)
        v_filename_filter = filter_file_for_split(split_method, SPLIT_NAME_VALIDATION)

    train_iter = load_dataset(ds_config,
                              filename_filter=t_filename_filter,
                              ds_split_name=SPLIT_NAME_TRAIN,
                              iterate_one_file=iterate_one_file,
                              rolling=rolling,
                              )
    val_iter = load_dataset(ds_config,
                            filename_filter=v_filename_filter,
                            ds_split_name=SPLIT_NAME_VALIDATION,
                            iterate_one_file=iterate_one_file,
                            rolling=rolling,
                            )
    return train_iter, val_iter


def load_dataset_no_split(ds_config: DatasetConfig,
                          filename_filter=None,
                          ):
    imeta = intervals_meta[ds_config.interval]
    rolling = imeta.get('rolling')
    iterate_one_file = imeta.get('iterate_one_file')
    batch_iter = load_dataset(ds_config,
                              filename_filter=filename_filter,
                              iterate_one_file=iterate_one_file,
                              rolling=rolling,
                              )
    return batch_iter


def load_and_group_files(dsc: DatasetConfig,
                         group_method,  # year/month/year-month
                         filename_filter=None,
                         ):
    dir = f'{dsc.base_dir}/{dsc.interval}/{dsc.symbol}'
    datapipe = dp.iter.FileLister(dir, masks='*.csv')
    if filename_filter is not None:
        datapipe = datapipe.filter(filter_fn=filename_filter)

    def extract(filename):
        base_name = os.path.basename(filename)
        # BTCUSDT-1d-2018-02.csv
        r = rf'.*-(20\d\d)-(\d\d).csv$'
        m = re.match(r, base_name)
        if m is None:
            return '-'
        m_year, m_month = m.groups()
        if group_method == 'year':
            return m_year
        if group_method == 'month':
            return m_month
        if group_method == 'year-month':
            return f'{m_year}-{m_month}'
        else:
            raise Exception(f'unknown filename group method: {group_method}')

    datapipe = datapipe.groupby(extract, keep_key=True)

    datapipe = datapipe.filter(filter_fn=lambda kv: kv[0] != '-')
    return datapipe


def load_dataset_for_files(ds_config: DatasetConfig, filenames, ds_split_name=None):
    dsc = ds_config
    datapipe = IterableWrapper(filenames)
    assert KlineSeqIterDataPipe
    datapipe = datapipe.load_kline_seq(dsc, ds_split_name)
    datapipe = datapipe.shuffle(buffer_size=dsc.shuffle_buffer)
    dl = DataLoader(dataset=datapipe,
                    batch_size=dsc.batch_size,
                    drop_last=True,
                    num_workers=0
                    )
    return dl


if __name__ == '__main__':
    set_print_options()

    kl_interval = '1d'
    market_base = '../../data/tm/market'
    ds_config = DatasetConfig(f'{market_base}/kline-basic',
                              symbol='ETHUSDT',
                              interval=kl_interval,
                              extra_intervals=extra_intervals_map.get(kl_interval),
                              )

    gf_iter = load_and_group_files(ds_config, 'year')

    for group_key, filenames in gf_iter:
        print(group_key, filenames)
        batch_iter = load_dataset_for_files(ds_config, filenames)
        for batch in batch_iter:
            print(batch)
            break

    # t_batch_iter, v_batch_iter = load_datasets(ds_config)
    #
    # last_batch = check_batch('train', t_batch_iter, check_count=1)
    # if last_batch is not None:
    #     seq = last_batch[-1].numpy()
    #     df = decode_seq(seq, ds_config.interval)
    #     print(df)
    #
    # print('======')

    # check_batch('validation', v_batch_iter)
