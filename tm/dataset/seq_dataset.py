import torch

from itertools import islice, takewhile, repeat
from tm.dataset.dataset_config import DatasetConfig
from tm.dataset.dataset_loader import load_datasets, load_dataset_no_split, load_and_group_files, load_dataset_for_files
from tm.dataset.batch_cache import load_cached_batches
from tm.common.utils import set_print_options


def split_every(n, iterable):
    iterator = iter(iterable)
    return takewhile(bool, (list(islice(iterator, n)) for _ in repeat(None)))


class SeqBatch:
    def __init__(self, input: torch.Tensor):
        self.input = input
        # (batch, seq_len, d_input_target)
        batch, seq_len, _ = input.shape
        target_fields_count = 3
        # self.src_mask = torch.ones(batch, 1, seq_len).int()
        self.src_mask = None
        self.src = input[:, :, 0:-target_fields_count]  # (batch, seq_len, d_input)
        # -3: b_ts, open(b_price): -2, f1_close: -1
        self.tgt_y = input[:, -1, -target_fields_count:]  # (batch,3)

        # o/h/l/c: 4/5/6/7
        # validate:
        # self.tgt_y[:, -1] = input[:, seq_len // 2, 4]  # (batch,2)
        # self.tgt_y[:, -1] = (input[:, seq_len // 2, 4] + src[:, 0, 4]) / 2

        # d_input = self.src.shape[-1]
        # self.src[:, :, range(4 + 4, d_input, 10)] = 0
        # self.src[:, :, range(4 + 5, d_input, 10)] = 0
        # self.src[:, :, range(4 + 6, d_input, 10)] = 0
        # test: mask buy_vol_ratio
        # self.src[:, :, range(4 + 7, d_input, 10)] = 0

        self.n_seqs = torch.tensor(batch)

    def mask_none(self):
        batch, seq_len, _ = self.src.shape
        return torch.ones(batch, 1, seq_len).int()  # (batch, 1, seq_len)


class SeqDataset:
    def __init__(self, ds_config: DatasetConfig, auto_setup=True):
        self.ds_config = ds_config
        self.train_batch_iter = None
        self.val_batch_iter = None
        if auto_setup:
            self.reset()

    def reset(self):
        dsc = self.ds_config
        if dsc.use_cached_batch:
            if dsc.cached_batch_base_dir is None:
                raise Exception('DatasetConfig.cached_batch_base_dir not set.')
            train_batch_iter, val_batch_iter = load_cached_batches(dsc)
        else:
            train_batch_iter, val_batch_iter = load_datasets(dsc)
        self.train_batch_iter = train_batch_iter
        self.val_batch_iter = val_batch_iter

    def train_step_batches_iter(self):
        for batch in self.train_batch_iter:
            yield SeqBatch(batch)

    def val_step_batches_iter(self):
        for batch in self.val_batch_iter:
            yield SeqBatch(batch)

    def train_val_steps_iter(self, train_batches_per_step, val_batches_per_step):
        t_batch_iter = self.train_step_batches_iter()
        v_batch_iter = self.val_step_batches_iter()
        t_steps_iter = split_every(train_batches_per_step, t_batch_iter)
        v_steps_iter = split_every(val_batches_per_step, v_batch_iter)
        return zip(t_steps_iter, v_steps_iter)

    def val_steps_iter(self, val_batches_per_step):
        v_data_iter = self.val_step_batches_iter()
        return split_every(val_batches_per_step, v_data_iter)

    def no_split_steps_iter(self, batches_per_step, filename_filter=None):
        batch_iter = load_dataset_no_split(self.ds_config, filename_filter=filename_filter)
        seq_batch_iter = map(SeqBatch, batch_iter)
        return split_every(batches_per_step, seq_batch_iter)

    def grouped_steps_iter(self,
                           group_method,  # year/month/year-month
                           batches_per_step,
                           filename_filter=None,
                           ):
        group_iter = load_and_group_files(self.ds_config, group_method, filename_filter)
        for group_key, filenames in group_iter:
            batch_iter = load_dataset_for_files(self.ds_config, filenames)
            seq_batch_iter = map(SeqBatch, batch_iter)
            steps_iter = split_every(batches_per_step, seq_batch_iter)
            yield group_key, steps_iter


if __name__ == '__main__':
    set_print_options()

    kl_interval = '1d'
    market_base = '../../data/tm/market'
    ds_config = DatasetConfig(f'{market_base}/basic/spot-kline',
                              symbol='ETHUSDT',
                              interval=kl_interval,
                              # extra_intervals=extra_intervals_map.get(kl_interval),
                              )

    ds = SeqDataset(ds_config, auto_setup=False)
    group_iter = ds.grouped_steps_iter('year', 10)
    for group_key, steps_iter in group_iter:
        print(f'group {group_key} ...')
        for i_step, batches_iter in enumerate(steps_iter):
            print(f'\tstep {i_step}')
            for i_batch, batch in enumerate(batches_iter):
                print(f'\t\tbatch {i_batch}, src: {batch.src.shape}')
