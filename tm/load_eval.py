import os
from datetime import date

import fire
import torch

from tm.model.evaluator import eval_state_dict, count_parameters, save_model_graph, export_model_onnx, eval_one_batch, \
    evaluate_and_log
from tm.dataset.seq_dataset import SeqDataset
from tm.common.utils import set_print_options, check_device


def main(
        cp_file,
        market_data_base_dir=None,
        cached_batch_base_dir=None,
        ds_split_method_name=None,
        use_cached_batch=None,
        batch_size: int = None,
        batches_per_step=None,
        use_cached_input=True,
        eval_batch=False,
        save_graph=False,
        export_onnx=False,
        show_state_dict=False,
        evaluate_epoch=True,
        evaluate_steps=None,
        group_method=None,  # year/month/year-month
        print_pred_detail=True,
        write_tensor_board=True,
        tensor_board_name=None,
):
    if cp_file is None:
        print('missing `cp_file`')
        return

    set_print_options()
    device = check_device()

    checkpoint_obj = torch.load(cp_file, map_location=device)

    model = checkpoint_obj['model']
    model_name = checkpoint_obj['model_name']
    model_config = checkpoint_obj['model_config']
    ds_config = checkpoint_obj['ds_config']
    encoder = checkpoint_obj.get('encoder')

    if not hasattr(ds_config, 'ds_split_method_name'):
        ds_config.ds_split_method_name = None

    if batch_size is not None:
        ds_config.seq_len = batch_size
    if market_data_base_dir is not None:
        ds_config.base_dir = f'{market_data_base_dir}/basic/spot-kline'
    if cached_batch_base_dir is not None:
        ds_config.cached_batch_base_dir = cached_batch_base_dir
    if use_cached_batch is not None:
        ds_config.use_cached_batch = use_cached_batch
    if ds_split_method_name is not None:
        ds_config.ds_split_method_name = ds_split_method_name
    if batches_per_step is not None:
        ds_config.val_batches_per_step = batches_per_step

    print(f'model name: {model_name}')

    out_dir = 'out'
    model_dir = 'med' if encoder else 'mdo'

    artifacts_dir = f'{out_dir}/artifacts/{model_dir}'

    def load_one_batch():
        cached_input_dir = f'{artifacts_dir}/input'
        os.makedirs(cached_input_dir, exist_ok=True)
        batch_file = f'{cached_input_dir}/{model_name}_input.pt'
        if use_cached_input:
            if os.path.exists(batch_file):
                return torch.load(batch_file)

        dataset = SeqDataset(ds_config)
        batches_iter = dataset.val_step_batches_iter()
        batch = next(batches_iter)
        torch.save(batch, batch_file)
        return batch

    if show_state_dict:
        eval_state_dict(model)

    pc = count_parameters(model)
    pcm = round(pc / 1000_000)
    print(f'trainable parameters: {pc} ({pcm}M)')

    batch = None
    if eval_batch or save_graph or export_onnx:
        batch = load_one_batch()

    if eval_batch:
        eval_one_batch(model, batch,
                       encoder=encoder,
                       device=device,
                       print_detail=print_pred_detail)
        print()

    if save_graph:
        tensor_board_model_dir = f'{out_dir}/runs/{model_dir}/models/{model_name}'
        save_model_graph(model, batch,
                         encoder=encoder,
                         device=device,
                         tensor_board_model_dir=tensor_board_model_dir)
        print()

    if export_onnx:
        onnx_file = f'{artifacts_dir}/{model_name}.onnx'
        export_model_onnx(model, batch, encoder=encoder, device=device, export_file=onnx_file)

    if evaluate_epoch:
        if tensor_board_name is None:
            tensor_board_name = f'runs-{ds_config.interval}-eval'
        date_str = date.today().isoformat()
        tensor_board_runs_dir = f'{out_dir}/{tensor_board_name}/{model_dir}/{date_str}'
        evaluate_and_log(model,
                         model_config,
                         ds_config,
                         steps=evaluate_steps,
                         encoder=encoder,
                         device=device,
                         model_name=model_name,
                         print_pred_detail=print_pred_detail,
                         tensor_board_runs_dir=tensor_board_runs_dir,
                         write_tensor_board=write_tensor_board,
                         chart_group_method=group_method,
                         )


if __name__ == "__main__":
    fire.Fire(main)
