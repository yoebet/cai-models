import numpy as np
import pandas as pd
import torch


def check_device():
    if torch.cuda.is_available():
        device_name = 'cuda:0'
    # elif torch.backends.mps.is_available():
    #     device_name = 'mps'
    else:
        device_name = 'cpu'
    device = torch.device(device_name)
    print(device)
    return device


def check_batch(name, batch_iter, check_count=2):
    last_batch = None
    n = 0
    for batch in batch_iter:
        print(batch)
        # (batch, seq_len, input_target)
        print(batch.shape)
        last_batch = batch
        n += 1
        if n == check_count:
            break
    print(f"{name}: {n = }")
    return last_batch


def cal_time_span_str(total_seconds):
    if total_seconds < 60:
        return f'{total_seconds} s'
    minutes, seconds = divmod(total_seconds, 60)
    if minutes < 60:
        return f'{minutes} m {seconds} s'
    hours, minutes = divmod(minutes, 60)
    return f'{hours} h {minutes} m {seconds} s'


def set_print_options(width=240, precision=4):
    pd.set_option('display.width', width)
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.max_colwidth', 160)
    pd.set_option('display.precision', precision)

    np.set_printoptions(linewidth=width, precision=precision)

    torch.set_printoptions(linewidth=width, precision=precision)
