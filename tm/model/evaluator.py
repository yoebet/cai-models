import os
import time
from datetime import datetime, timezone

import torch
from torch.utils.tensorboard import SummaryWriter
from tm.model.optimizer import build_val_loss_compute
from tm.model.trainer import run_step, run_batch, eval_predict, SRC_CP_IDX, gen_model_name, pred_percent_threshold, \
    print_val_detail
from tm.common.utils import cal_time_span_str
from tm.dataset.seq_dataset import SeqDataset, SeqBatch
from tm.dataset.dataset_config import DatasetConfig
from tm.model.model_config import ModelConfig


def eval_state_dict(model):
    sd = model.state_dict()
    c = 0
    i = 0
    for n, t in sd.items():
        a = t.numel()
        i += 1
        print(f'#{i} {n}:\t{a}')
        c += a
    print()
    print(c)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_input(batch: SeqBatch, encoder=False, device=None):
    src = batch.src.to(device, dtype=torch.float)
    src_mask = batch.mask_none().to(device, dtype=torch.float)
    if encoder:
        tgt = src.clone()
        tgt_mask = src_mask.clone()
        return (src, tgt, src_mask, tgt_mask)
    else:
        return (src, src_mask)


def save_model_graph(model, batch: SeqBatch, encoder=False, device=None, tensor_board_model_dir='runs/model'):
    writer = SummaryWriter(tensor_board_model_dir)
    input = get_model_input(batch, encoder=encoder, device=device)
    writer.add_graph(model, list(input))
    writer.flush()


def export_model_onnx(model, batch, encoder=False, device=None, export_file='exp/model.onnx'):
    input = get_model_input(batch, encoder=encoder, device=device)
    if encoder:
        input_names = ['src', 'tgt', 'src_mask', 'tgt_mask']
    else:
        input_names = ['src', 'src_mask']
    torch.onnx.export(model, input,
                      export_file,
                      # verbose=True,
                      input_names=input_names,
                      output_names=['o1'])


def eval_one_batch(model, batch: SeqBatch, encoder=False, device=None, print_detail=True):
    model.eval()
    start = time.time()
    loss_compute = build_val_loss_compute(model)
    loss, pred_y = run_batch(model, batch, loss_compute, encoder, device)
    # close: 7
    src_cp = batch.src[:, -1, SRC_CP_IDX]
    tgt_y = batch.tgt_y
    hit_count, s_pred_count, s_hit_count, detail = eval_predict(src_cp, tgt_y.to(src_cp.device),
                                                                pred_y.to(src_cp.device),
                                                                return_detail=print_detail)
    seq_count = batch.src.size(0)
    hit_percent = (hit_count / seq_count) * 100.0
    s_hit_percent = (s_hit_count / s_pred_count) * 100.0
    elapsed = time.time() - start
    seqs_per_sec = seq_count / elapsed

    if detail is not None:
        print()
        print(detail)
        print()

    print("Loss: %.6f, seq/s: %.2f,"
          " hit: %i/%i (%.1f%%) (%d/%d),"
          " s-hit: %i/%i (%.1f%%) (%d/%d)"
          % (loss, seqs_per_sec,
             hit_count, seq_count, hit_percent, hit_count, seq_count,
             s_hit_count, s_pred_count, s_hit_percent, s_hit_count, s_pred_count))


def evaluate_epoch(model,
                   steps_iter,
                   steps=None,
                   encoder=False,
                   val_loss_compute=None,
                   device=None,
                   print_pred_detail=False,
                   tensor_writer: SummaryWriter = None,
                   text_tag='step',
                   ):
    acc_hc = 0
    acc_seqs_count = 0
    acc_s_hc = 0
    acc_s_pred_count = 0
    actual_steps = 0
    for i_step, batches_iter in enumerate(steps_iter):
        print('------')
        global_step = i_step
        print("step %d:" % i_step)

        model.eval()
        v_batches, v_loss, v_seqs_per_sec, val_detail = run_step(batches_iter,
                                                                 model,
                                                                 val_loss_compute,
                                                                 encoder=encoder,
                                                                 device=device,
                                                                 print_pred_detail=print_pred_detail,
                                                                 )
        if v_batches == 0:
            continue

        print_val_detail(1, 1, v_batches,
                         v_loss, v_seqs_per_sec, val_detail,
                         global_step, tensor_writer=tensor_writer,
                         text_tag=text_tag,
                         )

        val_hc, val_seqs_count, val_s_hc, val_s_pred_count, _pred_detail = val_detail
        acc_hc += val_hc
        acc_seqs_count += val_seqs_count
        acc_s_hc += val_s_hc
        acc_s_pred_count += val_s_pred_count
        acc_hit_rate = acc_hc / acc_seqs_count if acc_seqs_count > 0 else 0.5
        acc_s_hit_rate = acc_s_hc / acc_s_pred_count if acc_s_pred_count > 0 else 0.5
        acc_info = f'(ACC) hp {acc_hit_rate:.1%}' \
                   f' ({acc_hc}/{acc_seqs_count}),' \
                   f' s-hp {acc_s_hit_rate:.1%}' \
                   f' ({acc_s_hc}/{acc_s_pred_count})'
        print(acc_info)
        tensor_writer.add_text(text_tag, acc_info, global_step, time.time())

        actual_steps = i_step + 1
        if steps is not None and actual_steps >= steps:
            break

    return actual_steps


def evaluate_and_log(model,
                     mc: ModelConfig,
                     dsc: DatasetConfig,
                     steps=None,
                     encoder=False,
                     device=None,
                     model_name=None,
                     print_pred_detail=False,
                     write_tensor_board=True,
                     tensor_board_runs_dir='runs/',
                     chart_group_method=None,  # year/month/year-month
                     ):
    start = time.time()
    if model_name is None:
        model_name = gen_model_name(mc, dsc, encoder=encoder)
    tensor_board_dir = os.path.join(tensor_board_runs_dir, model_name)
    tensor_writer = SummaryWriter(tensor_board_dir) if write_tensor_board else None
    print('evaluate ...')
    print(f'pred threshold {pred_percent_threshold}%')
    if tensor_writer is not None:
        time_str = datetime.fromtimestamp(start, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        tensor_writer.add_text('step',
                               f'{time_str}, evaluate ...\n\n'
                               f'model: {model_name}\n\n'
                               f'pred threshold {pred_percent_threshold}%',
                               0,
                               start)
        tensor_writer.flush()

    val_loss_compute = build_val_loss_compute(model)

    dataset = SeqDataset(dsc)
    val_bpr = dsc.val_batches_per_step

    epoch_start = time.time()

    actual_steps = 0

    if chart_group_method is None:
        v_steps_iter = dataset.no_split_steps_iter(val_bpr)

        actual_steps = evaluate_epoch(model,
                                      v_steps_iter,
                                      steps=steps,
                                      encoder=encoder,
                                      val_loss_compute=val_loss_compute,
                                      device=device,
                                      print_pred_detail=print_pred_detail,
                                      tensor_writer=tensor_writer,
                                      )
    else:
        group_iter = dataset.grouped_steps_iter(chart_group_method, val_bpr)

        for group_key, steps_iter in group_iter:
            tb_group_dir = os.path.join(tensor_board_dir, chart_group_method, group_key)
            group_tensor_writer = SummaryWriter(tb_group_dir) if write_tensor_board else None

            print(f'===')
            print(f'group {group_key} ...')
            steps0 = evaluate_epoch(model,
                                    steps_iter,
                                    steps=steps,
                                    encoder=encoder,
                                    val_loss_compute=val_loss_compute,
                                    device=device,
                                    print_pred_detail=print_pred_detail,
                                    tensor_writer=group_tensor_writer,
                                    text_tag=group_key,
                                    )
            actual_steps += steps0

    val_batches = actual_steps * val_bpr
    batch_size = dsc.batch_size
    print()
    print(
        f'valid: {actual_steps} steps, ~{val_batches} batches, ~{val_batches * batch_size} seqs')
    print('train cost: ', round(time.time() - epoch_start))

    seq_len = dsc.seq_len
    ts = time.time()
    total_cost = round(ts - start)
    time_span_str = cal_time_span_str(total_cost)
    print('======')
    print(f'batches per step {val_bpr}, {batch_size = }, {seq_len = }')
    print(f'total steps: {actual_steps}')
    print('total cost: ', time_span_str)
    print('------')

    if tensor_writer is not None:
        time_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        done_info = f'{time_str}, done. cost {time_span_str}'
        tensor_writer.add_text('step', done_info, actual_steps, ts)
        tensor_writer.flush()

    return actual_steps
