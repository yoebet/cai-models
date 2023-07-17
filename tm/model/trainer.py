import os
import time
from datetime import datetime, timezone
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from tm.common.utils import cal_time_span_str
from tm.dataset.seq_dataset import SeqDataset, SeqBatch
from tm.dataset.dataset_config import DatasetConfig
from tm.model.model_config import ModelConfig
from tm.model.optimizer import build_train_val_loss_compute

SRC_CP_IDX = 7
pred_percent_threshold = 0.2  # 0.2%
classify_centre_percent = 0.2  # 0.2%

pred_detail_columns = ['time', 'curr_p', 'tgt_p', 'pred_p', 'hit?', 's_hit?',
                       'pred_percent', 'tgt_percent', 'percent_diff']

c_pred_detail_columns = ['time', 'curr_p', 'tgt_p', 'tgt_percent', 'label', 'pred', 'hit?', 's_hit?']


def to_time_str(d):
    dt = datetime.utcfromtimestamp(round(d))
    return dt.isoformat(sep=' ', timespec='minutes')


def eval_predict_regression(src_cp: torch.Tensor,  # (batch,)
                            tgt_y_a: torch.Tensor,  # (batch,x)
                            pred_y_bp: torch.Tensor,  # (batch,)
                            return_detail=False,
                            ):
    tgt_y_bp = tgt_y_a[:, -1]
    tgt_y = tgt_y_bp / src_cp
    pred_y = pred_y_bp / src_cp

    tgt_y_ud = tgt_y >= 1.0
    pred_ud = pred_y >= 1.0
    hit = tgt_y_ud == pred_ud
    hit_count = torch.sum(hit)

    s_threshold = pred_percent_threshold / 100
    s_pred_y = (pred_y - 1).abs() > s_threshold
    s_pred_count = torch.sum(s_pred_y)
    s_hit = hit[s_pred_y]
    s_hit_count = torch.sum(s_hit)

    detail = None

    if return_detail:
        bp = tgt_y_a[:, -2]
        ts = tgt_y_a[:, -3]
        close_p = src_cp * bp
        tgt_p = tgt_y_bp * bp
        pred_p = pred_y_bp * bp
        tgt_y_percent = (tgt_y - 1) * 100
        pred_y_percent = (pred_y - 1) * 100
        percent_diff = pred_y_percent - tgt_y_percent
        detail_t = torch.stack((ts, close_p, tgt_p, pred_p, hit, s_pred_y,
                                pred_y_percent, tgt_y_percent, percent_diff), dim=1)
        detail = pd.DataFrame(detail_t.numpy(), columns=pred_detail_columns)

        detail['time'] = detail['time'].apply(to_time_str)

    return hit_count, s_pred_count, s_hit_count, detail


def eval_predict_classification(src_cp: torch.Tensor,  # (batch,)
                                tgt_y_a: torch.Tensor,  # (batch,x)
                                pred_y: torch.Tensor,  # (batch,) # dtype=long
                                label_y: torch.Tensor,  # (batch,) # dtype=long
                                return_detail=False,
                                ):
    tgt_y_bp = tgt_y_a[:, -1]
    tgt_y = tgt_y_bp / src_cp
    hit = pred_y == label_y
    hit_count = torch.sum(hit)
    s_mask = pred_y != 1
    s_pred_count = torch.sum(s_mask)

    s_hit = hit.clone()
    s_hit[~s_mask] = 0

    s_hit_count = torch.sum(s_hit)
    detail = None

    if return_detail:
        bp = tgt_y_a[:, -2]
        ts = tgt_y_a[:, -3]
        close_p = src_cp * bp
        tgt_p = tgt_y_bp * bp
        tgt_y_percent = (tgt_y - 1) * 100
        detail_t = torch.stack((ts, close_p, tgt_p, tgt_y_percent, label_y, pred_y, hit, s_hit), dim=1)
        detail = pd.DataFrame(detail_t.numpy(), columns=c_pred_detail_columns)

        def cn(c):
            return ['<', '=', '>'][round(c)]

        detail['time'] = detail['time'].apply(to_time_str)
        detail['pred'] = detail['pred'].apply(cn)
        detail['label'] = detail['label'].apply(cn)

    return hit_count, s_pred_count, s_hit_count, detail


def eval_predict(src_cp: torch.Tensor,  # (batch,)
                 tgt_y_a: torch.Tensor,  # (batch,x)
                 pred_y: torch.Tensor,  # (batch,)
                 label_y: torch.Tensor = None,  # (batch,)
                 return_detail=False,
                 classify=False,
                 ):
    if classify:
        return eval_predict_classification(src_cp, tgt_y_a, pred_y, label_y, return_detail=return_detail)

    return eval_predict_regression(src_cp, tgt_y_a, pred_y, return_detail=return_detail)


def run_batch(model,
              batch,
              loss_compute,
              encoder=False,
              device=None
              ):
    dtype = torch.float
    src = batch.src.to(device, dtype=dtype)
    src_mask = batch.src_mask.to(device, dtype=dtype) if batch.src_mask is not None else None
    if encoder:
        tgt = src.clone().to(device, dtype=dtype)
        tgt_mask = src_mask.clone().to(device, dtype=dtype) if src_mask is not None else None
        out = model.forward(src, tgt, src_mask, tgt_mask)
    else:
        out = model.forward(src, src_mask)
    out = loss_compute.generator(out)  # (batch,) or (batch,n)
    pred_y = out.clone().detach()
    # f1_close: -1
    label_y = batch.tgt_y[:, -1]
    if loss_compute.generator.is_classifier:
        src_cp = batch.src[:, -1, SRC_CP_IDX]
        pr = label_y / src_cp
        cr = classify_centre_percent / 100
        label_y = torch.ones(pr.size(0), dtype=torch.long)
        label_y[pr < (1 - cr)] = 0
        label_y[pr > (1 + cr)] = 2
        pred_y = torch.argmax(pred_y, -1)
    else:
        label_y = label_y.to(dtype=dtype)
    loss = loss_compute(out, label_y.to(out.device), batch.n_seqs.to(out.device))
    return loss, label_y, pred_y


def run_step(data_iter,
             model,
             loss_compute,
             encoder=False,
             device=None,
             print_pred_detail=False,
             ):
    start = time.time()
    total_batches = 0
    total_seqs = 0
    total_loss = 0
    val_hit_count = 0
    val_seqs_count = 0
    val_s_pred_count = 0
    val_s_hit_count = 0
    pred_detail = None
    last_eval_predict_input = None
    is_classifier = loss_compute.generator.is_classifier
    for i, batch in enumerate(data_iter):
        seq_count = batch.src.size(0)
        loss, label_y, pred_y = run_batch(model, batch, loss_compute, encoder, device)
        tgt_y = batch.tgt_y
        total_loss += loss
        total_seqs += batch.n_seqs
        total_batches += 1
        if not model.training:
            src_cp = batch.src[:, -1, SRC_CP_IDX]
            hit_count, s_pred_count, s_hit_count, _detail = eval_predict(src_cp,
                                                                         tgt_y,
                                                                         pred_y,
                                                                         label_y=label_y,
                                                                         return_detail=False,
                                                                         classify=is_classifier
                                                                         )
            last_eval_predict_input = (src_cp, tgt_y, label_y, pred_y)
            val_hit_count += hit_count
            val_seqs_count += seq_count
            val_s_pred_count += s_pred_count
            val_s_hit_count += s_hit_count

    if not model.training and total_batches > 0 and print_pred_detail:
        src_cp, tgt_y, label_y, pred_y = last_eval_predict_input
        hit_count, s_pred_count, s_hit_count, detail = eval_predict(src_cp,
                                                                    tgt_y,
                                                                    pred_y,
                                                                    label_y=label_y,
                                                                    return_detail=True,
                                                                    classify=is_classifier
                                                                    )
        if detail is not None:
            print()
            print(detail)
            print()
            pred_detail = detail

    if total_batches == 0:
        return 0, 0, 0, None

    elapsed = time.time() - start
    loss_per_seq = total_loss / total_seqs
    seqs_per_sec = total_seqs / elapsed

    val_detail = val_hit_count, val_seqs_count, val_s_hit_count, val_s_pred_count, pred_detail

    return total_batches, loss_per_seq, seqs_per_sec, val_detail


def train_epoch(model,
                vt_steps_iter,
                steps=None,
                epochs=1,
                current_epoch=0,
                global_steps=0,
                encoder=False,
                optimizer=None,
                train_loss_compute=None,
                val_loss_compute=None,
                device=None,
                print_pred_detail=False,
                tensor_writer: SummaryWriter = None,
                ):
    actual_steps = 0
    for i_step, (t_batches_iter, v_batches_iter) in enumerate(vt_steps_iter):
        print('------')
        global_step = global_steps + i_step
        n_epoch = current_epoch + 1

        if epochs is not None and epochs > 1:
            print("Epoch %d/%d, step %d:" % (n_epoch, epochs, i_step))
        else:
            print("step %d:" % i_step)
        model.train()
        t_batches, t_loss, t_seqs_per_sec, _val_detail = run_step(t_batches_iter,
                                                                  model,
                                                                  train_loss_compute,
                                                                  encoder=encoder,
                                                                  device=device,
                                                                  )
        if t_batches == 0:
            continue

        print("Train, Loss: %.8f, seq/s: %.2f" % (t_loss, t_seqs_per_sec))
        lr = optimizer.param_groups[0]["lr"]
        print(f'lr: {lr:.8f}, train batches: {t_batches}')

        if tensor_writer is not None:
            ts = time.time()
            tensor_writer.add_scalar('LR', lr, global_step, ts)
            tensor_writer.add_scalar('Loss (T)', t_loss, global_step, ts)

        model.eval()
        v_batches, v_loss, v_seqs_per_sec, val_detail = run_step(v_batches_iter,
                                                                 model,
                                                                 val_loss_compute,
                                                                 encoder=encoder,
                                                                 device=device,
                                                                 print_pred_detail=print_pred_detail,
                                                                 )
        if v_batches == 0:
            continue

        print_val_detail(epochs, n_epoch, v_batches,
                         v_loss, v_seqs_per_sec, val_detail,
                         global_step, tensor_writer=tensor_writer,
                         )

        actual_steps = i_step + 1
        if steps is not None and actual_steps >= steps:
            break

    return actual_steps


def print_val_detail(epochs, n_epoch, v_batches,
                     v_loss, v_seqs_per_sec, val_detail,
                     global_step,
                     tensor_writer=None,
                     text_tag='step',
                     ):
    val_hit_count, val_seqs_count, val_s_hit_count, val_s_pred_count, pred_detail = val_detail

    val_hit_rate = val_hit_count / val_seqs_count if val_seqs_count > 0 else 0.5
    val_s_hit_rate = val_s_hit_count / val_s_pred_count if val_s_pred_count > 0 else 0.5
    val_hit_percent = val_hit_rate * 100.0
    val_s_hit_percent = val_s_hit_rate * 100.0

    print("Valid, Loss: %.8f, seq/s: %.2f" % (v_loss, v_seqs_per_sec))
    ts = time.time()
    time_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S")
    step_info = f'{time_str},'
    if epochs > 1:
        step_info += f' Epoch {n_epoch}/{epochs},'
    step_info += f' step {global_step},' \
                 f' batches {v_batches},' \
                 f' hp {val_hit_rate:.1%}' \
                 f' ({val_hit_count}/{val_seqs_count}),' \
                 f' s-hp {val_s_hit_rate:.1%}' \
                 f' ({val_s_hit_count}/{val_s_pred_count})'
    print(step_info)

    if tensor_writer is not None:
        tensor_writer.add_scalar('Loss (V)', v_loss, global_step, ts)
        tensor_writer.add_scalar('HP (%)', val_hit_percent, global_step, ts)
        tensor_writer.add_scalar('HP-S (%)', val_s_hit_percent, global_step, ts)

        if pred_detail is not None:
            dfs = pred_detail.to_markdown()
            dfs = dfs + '\n\n' + step_info
            tensor_writer.add_text(text_tag, dfs, global_step, ts)
        else:
            tensor_writer.add_text(text_tag, step_info, global_step, ts)
        tensor_writer.flush()


def gen_model_name(
        mc: ModelConfig,
        dsc: DatasetConfig,
        encoder=False,
        postfix=None,
):
    # mt = 'edm' if encoder else 'dom'
    bt = 'x2' if encoder else ''
    coin = dsc.symbol.lower().replace('usdt', '')
    input_props = f'{coin}-{dsc.interval}-sl{dsc.seq_len}-di{mc.d_input}'
    mt = f'c{mc.classify_n}' if mc.classify else 'r'
    postfix = f'-{postfix}' if postfix is not None else ''
    return f'{input_props}-dm{mc.d_model}-h{mc.heads}-b{mc.blocks}{bt}-df{mc.d_ff}-dg{mc.d_gen_ff}-{mt}{postfix}'


def train_and_log(model,
                  mc: ModelConfig,
                  dsc: DatasetConfig,
                  epochs=1,
                  steps=None,
                  encoder=False,
                  device=None,
                  model_name=None,
                  print_pred_detail=False,
                  write_tensor_board=True,
                  tensor_board_runs_dir='runs/',
                  save_cp_every_epochs=1,
                  checkpoint_dir=None,
                  start_epoch=0,
                  start_global_steps=0,
                  optimizer_state=None,
                  model_opt_state=None,
                  ):
    start = time.time()
    if model_name is None:
        model_name = gen_model_name(mc, dsc, encoder=encoder)
    tensor_board_dir = os.path.join(tensor_board_runs_dir, model_name)
    tensor_writer = SummaryWriter(tensor_board_dir) if write_tensor_board else None

    print(f'pred threshold {pred_percent_threshold}%')
    if tensor_writer is not None:
        time_str = datetime.fromtimestamp(start, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        tensor_writer.add_text('step',
                               f'{time_str}, model: {model_name}, pred threshold {pred_percent_threshold}%',
                               0,
                               start)
        tensor_writer.flush()

    optimizer, train_loss_compute, \
        val_loss_compute = build_train_val_loss_compute(model, mc,
                                                        optimizer_state=optimizer_state,
                                                        noam_state=model_opt_state)

    model_opt = train_loss_compute.opt

    dataset = SeqDataset(dsc)
    train_bpr = dsc.train_batches_per_step
    val_bpr = dsc.val_batches_per_step

    if epochs > 1:
        if steps is not None:
            print(f'ignore `steps` while `epochs` > 1')
        steps = None

    global_steps = start_global_steps

    for epoch in range(start_epoch, epochs):
        print('======')
        epoch_start = time.time()

        if epoch > start_epoch:
            dataset.reset()
        vt_steps_iter = dataset.train_val_steps_iter(train_bpr,
                                                     val_bpr)

        actual_steps = train_epoch(model,
                                   vt_steps_iter,
                                   steps=steps,
                                   epochs=epochs,
                                   current_epoch=epoch,
                                   global_steps=global_steps,
                                   encoder=encoder,
                                   optimizer=optimizer,
                                   train_loss_compute=train_loss_compute,
                                   val_loss_compute=val_loss_compute,
                                   device=device,
                                   print_pred_detail=print_pred_detail,
                                   tensor_writer=tensor_writer,
                                   )

        train_batches = actual_steps * train_bpr
        val_batches = actual_steps * val_bpr
        batch_size = dsc.batch_size
        print()
        print(
            f'train: {actual_steps} steps, ~{train_batches} batches, ~{train_batches * batch_size} seqs')
        print(
            f'valid: {actual_steps} steps, ~{val_batches} batches, ~{val_batches * batch_size} seqs')
        print('train cost: ', round(time.time() - epoch_start))

        global_steps += actual_steps

        if steps is not None:
            continue

        if save_cp_every_epochs is not None and checkpoint_dir is not None:
            last_epoch = (epoch + 1) == epochs
            if not last_epoch and (epoch + 1) % save_cp_every_epochs != 0:
                continue
            cp_filename = f'{model_name}_epoch-{epoch + 1}.pt'
            save_path = os.path.join(checkpoint_dir, cp_filename)
            save_checkpoint(save_path, model, model_name, mc, dsc,
                            plan_epochs=epochs,
                            run_epochs=epoch + 1,
                            run_steps=global_steps,
                            encoder=encoder,
                            optimizer_state=optimizer.state_dict(),
                            model_opt_state=model_opt.state_dict(),
                            )

    batch_size = dsc.batch_size
    seq_len = dsc.seq_len
    ts = time.time()
    total_cost = round(ts - start)
    time_span_str = cal_time_span_str(total_cost)
    print('======')
    print(f'config: batches per step {train_bpr}/{val_bpr}, {batch_size = }, {seq_len = }')
    if start_epoch > 0:
        print(f'total epochs: {epochs - start_epoch} ({start_epoch} - {epochs})')
    else:
        print(f'total epochs: {epochs}')
    if start_global_steps > 0:
        print(f'total steps: {global_steps - start_global_steps} ({start_global_steps} - {global_steps})')
    else:
        print(f'total steps: {global_steps}')
    print('total cost: ', time_span_str)
    print('------')

    if tensor_writer is not None:
        time_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        tensor_writer.add_text('step', f'{time_str}, done. {epochs} epochs, cost {time_span_str}', global_steps, ts)
        tensor_writer.flush()

    return optimizer, model_opt, global_steps


def save_checkpoint(save_path,
                    model,
                    model_name,
                    mc: ModelConfig,
                    dsc: DatasetConfig,
                    plan_epochs=None,
                    run_epochs=None,
                    run_steps=None,
                    encoder=False,
                    optimizer_state=None,
                    model_opt_state=None,
                    ):
    cp_dir = os.path.dirname(save_path)
    os.makedirs(cp_dir, exist_ok=True)
    obj = {'model': model,
           'model_name': model_name,
           'model_config': mc,
           'ds_config': dsc,
           'plan_epochs': plan_epochs,
           'run_epochs': run_epochs,
           'run_steps': run_steps,
           'encoder': encoder,
           'optimizer_state': optimizer_state,
           'model_opt_state': model_opt_state,
           }
    torch.save(obj, save_path)
