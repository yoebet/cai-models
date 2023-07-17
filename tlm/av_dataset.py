import os, re
from itertools import islice, takewhile, repeat
import torch
import torchdata.datapipes as dp
from torch.utils.data import DataLoader
from dataset_config import DatasetConfig, KlineDatasetConfig
import news_vec_iter_dp


def split_every(n, iterable):
    iterator = iter(iterable)
    return takewhile(bool, (list(islice(iterator, n)) for _ in repeat(None)))


def load_dataset(ds_config: DatasetConfig, filename_filter=None, name=None):
    datapipe = dp.iter.FileLister(ds_config.news_meta_dir,
                                  non_deterministic=True,
                                  recursive=True,
                                  masks='*.jsonl'
                                  )
    if filename_filter is not None:
        datapipe = datapipe.filter(filter_fn=filename_filter)
    datapipe = datapipe.load_news_vec(ds_config, name)
    # datapipe = datapipe.shuffle(buffer_size=ds_config.shuffle_buffer)
    dl = DataLoader(dataset=datapipe,
                    # batch_size=ds_config.batch_size,
                    drop_last=True,
                    num_workers=0
                    )
    return dl


def filter_file_by_pattern(pat):
    def fp(filename):
        # fn = os.path.basename(filename)
        # return '-2022-' in fn
        if '/2017/' in filename:
            return False
        return re.search(pat, filename) is not None

    return fp


def load_datasets(ds_config: DatasetConfig):
    train_iter = load_dataset(ds_config,
                              filename_filter=filter_file_by_pattern(r'/20[12]\d-0[^39]-'), name='train')
    val_iter = load_dataset(ds_config,
                            filename_filter=filter_file_by_pattern(r'/20[12]\d-0[39]-'), name='valid')
    return train_iter, val_iter


class AvBatch:
    def __init__(self, src: torch.Tensor):
        # (batch, seq_len, d_input_target)
        batch, seq_len, _ = src.shape


class AvDataset:
    def __init__(self, ds_config: DatasetConfig):
        self.ds_config = ds_config
        self.train_batch_iter = None
        self.val_batch_iter = None
        self.reset()

    def reset(self):
        train_batch_iter, val_batch_iter = load_datasets(self.ds_config)
        self.train_batch_iter = train_batch_iter
        self.val_batch_iter = val_batch_iter

    def train_step_batches_iter(self):
        for batch in self.train_batch_iter:
            yield AvDataset(batch)

    def val_step_batches_iter(self):
        for batch in self.val_batch_iter:
            yield AvDataset(batch)

    def train_val_batches_iter(self, train_batches_per_step, val_batches_per_step):
        t_data_iter = self.train_step_batches_iter()
        v_data_iter = self.val_step_batches_iter()
        t_batches_iter = split_every(train_batches_per_step, t_data_iter)
        v_batches_iter = split_every(val_batches_per_step, v_data_iter)
        return zip(t_batches_iter, v_batches_iter)


def check_batch(name, batch_iter):
    n = 0
    for batch in batch_iter:
        print(batch)
        # (batch, seq_len, input_target)
        # print(batch.shape)
        n += 1
        if n == 2:
            break
    print(f"{name}: {n = }")


if __name__ == '__main__':
    torch.set_printoptions(precision=3)
    base_dir = '../data/tm'
    kl_config = KlineDatasetConfig(f'{base_dir}/market/basic/spot-kline')
    ds_config = DatasetConfig(kl_config,
                              news_meta_dir=f'{base_dir}/news_meta',
                              news_vec_dir=f'{base_dir}/news_vec/instructor_xl',
                              )
    t_batch_iter, v_batch_iter = load_datasets(ds_config)

    check_batch('train', t_batch_iter)

    print('======')

    check_batch('validate', v_batch_iter)
