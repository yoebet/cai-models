from datetime import datetime

import fire
import torch
from tm.dataset.dataset_config import DatasetConfig, extra_intervals_map
from tm.dataset.dataset_loader import load_datasets
from tm.dataset.batch_cache import build_batch_tensors, load_cached_batches, gen_tensors_base_path
from tm.common.utils import set_print_options, check_batch
from tm.dataset.kl_seq_iter_dp import decode_seq


def main(symbol='ETHUSDT',
         kl_interval='4h',
         seq_len=32,
         batch_size=10,
         train_batches_per_step=180,
         val_batches_per_step=30,
         market_data_base_dir='../data/tm/market',
         cached_batch_base_dir='../data/tm/batch_tensor',
         ds_split_method_name=None,
         batches_per_tensor_file=100,
         action='build_tensors'
         ):
    if action is None:
        print(f'missing `action`')

    print(datetime.now())
    set_print_options()

    symbol = symbol.upper()
    dsc = DatasetConfig(
        base_dir=f'{market_data_base_dir}/kline-basic',
        cached_batch_base_dir=cached_batch_base_dir,
        symbol=symbol,
        interval=kl_interval,
        extra_intervals=extra_intervals_map.get(kl_interval),
        seq_len=seq_len,
        batch_size=batch_size,
        train_batches_per_step=train_batches_per_step,
        val_batches_per_step=val_batches_per_step,
        ds_split_method_name=ds_split_method_name,
    )

    if action == 'build_tensors':
        build_batch_tensors(dsc, batches_per_file=batches_per_tensor_file)
    elif action == 'check_cached_batch_file':
        tensor_base_path = gen_tensors_base_path(dsc)
        tt = torch.load(tensor_base_path / 'train/00.pt')
        assert type(tt) == list
        # len(tt) = batches_per_tensor_file
        batch_tensor = tt[0]
        # print(batch_tensor)
        print(batch_tensor.shape)  # (batch_size, seq_len, d_input_target)
        seq = batch_tensor[-1].numpy()
        df = decode_seq(seq, dsc.interval)
        print(df)
    elif action == 'check_cached_batch':
        t_batch_iter, v_batch_iter = load_cached_batches(dsc)
        last_batch=check_batch('train', t_batch_iter)
        if last_batch is not None:
            seq = last_batch[-1].numpy()
            df = decode_seq(seq, dsc.interval)
            print(df)
        print('======')
        # check_batch('validate', v_batch_iter)
    elif action == 'check_csv_batch':
        t_batch_iter, v_batch_iter = load_datasets(dsc)
        check_batch('train', t_batch_iter)
        print('======')
        # check_batch('validate', v_batch_iter)


if __name__ == '__main__':
    fire.Fire(main)
