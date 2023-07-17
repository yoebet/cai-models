import os
import pathlib
import time
import shutil

import torch
import torchdata.datapipes as dp
from torchdata.datapipes.iter import IterableWrapper
from tm.dataset.dataset_config import DatasetConfig, SPLIT_NAME_TRAIN, SPLIT_NAME_VALIDATION
from tm.dataset.dataset_loader import load_datasets
from tm.dataset.dataset_split import DEFAULT_SPLIT_METHOD
from tm.common.utils import set_print_options, check_batch


def cache_tensors(batch_iter,
                  tensor_base_path,
                  split_name,
                  batches_per_file=100,
                  ):
    start = time.time()
    tensor_dp = IterableWrapper(batch_iter)
    tensor_dp = tensor_dp.batch(batches_per_file, wrapper_class=list)

    files_count = 0
    batches_count = 0
    tensor_path = pathlib.Path(tensor_base_path, split_name)
    if os.path.exists(tensor_path):
        shutil.rmtree(tensor_path)
    os.makedirs(tensor_path)
    for i, batch_list in enumerate(tensor_dp):
        # batch_list = [b.to(dtype=torch.float32) for b in batch_list]
        fname = str(i).rjust(2, '0') + '.pt'
        torch.save(batch_list, tensor_path / fname)
        files_count += 1
        batches_count += len(batch_list)

    print(
        f'build and cache `{split_name}` tensors done, cost {round(time.time() - start)}s.\n'
        f'files: {files_count}, batches: {batches_count}\n'
        f'---')

    return files_count, batches_count


def gen_tensors_base_path(ds_config: DatasetConfig):
    dsc = ds_config
    split_method = dsc.ds_split_method_name
    if split_method is None:
        split_method = DEFAULT_SPLIT_METHOD
    return pathlib.Path(ds_config.cached_batch_base_dir,
                        split_method,
                        f'{dsc.interval}-{dsc.symbol}',
                        f'sl{dsc.seq_len}-bs{dsc.batch_size}'
                        )


def build_batch_tensors(ds_config: DatasetConfig,
                        batches_per_file=100,
                        ):
    train_batch_iter, val_batch_iter = load_datasets(ds_config)

    tensor_base_path = gen_tensors_base_path(ds_config)
    os.makedirs(tensor_base_path, exist_ok=True)

    cache_tensors(train_batch_iter,
                  tensor_base_path,
                  SPLIT_NAME_TRAIN,
                  batches_per_file=batches_per_file
                  )
    cache_tensors(val_batch_iter,
                  tensor_base_path,
                  SPLIT_NAME_VALIDATION,
                  batches_per_file=batches_per_file
                  )


def load_cached_tensors(ds_config: DatasetConfig,
                        split_name
                        ):
    tensor_base_path = gen_tensors_base_path(ds_config)
    tensor_path = os.path.join(tensor_base_path, split_name)
    datapipe = dp.iter.FileLister(tensor_path,
                                  non_deterministic=True,
                                  masks='*.pt',
                                  )
    datapipe = datapipe.map(torch.load)
    datapipe = datapipe.unbatch()
    datapipe = datapipe.shuffle(buffer_size=ds_config.shuffle_buffer)
    return datapipe


def load_cached_batches(ds_config: DatasetConfig):
    train_iter = load_cached_tensors(ds_config, split_name=SPLIT_NAME_TRAIN)
    val_iter = load_cached_tensors(ds_config, split_name=SPLIT_NAME_VALIDATION)
    return train_iter, val_iter


if __name__ == '__main__':
    set_print_options()

    kl_interval = '4h'
    market_base = '../../data/tm/market'
    ds_config = DatasetConfig(f'{market_base}/kline-basic',
                              symbol='ETHUSDT',
                              interval=kl_interval,
                              seq_len=32,
                              batch_size=10,
                              cached_batch_base_dir='../../data/tm/batch_tensor',
                              use_cached_batch=True,
                              )

    t_batch_iter, v_batch_iter = load_cached_batches(ds_config)

    check_batch('train', t_batch_iter)

    print('======')

    # check_batch('validation', v_batch_iter)
