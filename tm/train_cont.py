from datetime import datetime, date

import fire
import torch

from tm.model.trainer import train_and_log, save_checkpoint
from tm.common.utils import set_print_options, check_device


def main(
        cp_file,
        epochs: int,
        model_type=None,
        market_data_base_dir=None,
        cached_batch_base_dir=None,
        ds_split_method_name=None,
        use_cached_batch=True,
        print_pred_detail=True,
        write_tensor_board=True,
        tensor_board_name=None,
        save_cp_every_epochs=4,
):
    if cp_file is None:
        print('missing `cp_file`')
        return

    print(datetime.now())

    set_print_options()

    device = check_device()

    checkpoint_obj = torch.load(cp_file, map_location=device)

    model = checkpoint_obj['model']
    model_name = checkpoint_obj['model_name']
    model_config = checkpoint_obj['model_config']
    ds_config = checkpoint_obj['ds_config']
    # plan_epochs = checkpoint_obj['plan_epochs']
    run_epochs = checkpoint_obj['run_epochs']
    run_steps = checkpoint_obj['run_steps']
    optimizer_state = checkpoint_obj.get('optimizer_state')
    model_opt_state = checkpoint_obj.get('model_opt_state')
    encoder = checkpoint_obj.get('encoder')
    if encoder is None:
        encoder = False

    if model_type is not None:
        if model_type == 'decoder_only':
            encoder = False
        else:
            encoder = True

    out_dir = 'out'
    model_dir = 'med' if encoder else 'mdo'

    artifacts_dir = f'{out_dir}/artifacts/{model_dir}'

    if market_data_base_dir is not None:
        ds_config.base_dir = f'{market_data_base_dir}/kline-basic'
    if cached_batch_base_dir is not None:
        ds_config.cached_batch_base_dir = cached_batch_base_dir
    if use_cached_batch is not None:
        ds_config.use_cached_batch = use_cached_batch
    if ds_split_method_name is not None:
        ds_config.ds_split_method_name = ds_split_method_name

    if epochs <= run_epochs:
        print(f'already run {run_epochs} epochs.')
        return

    print(f'model name: {model_name}')
    print(f'batch size: {ds_config.batch_size},'
          f' batches per step: {ds_config.train_batches_per_step}/{ds_config.val_batches_per_step}')
    print(f'start epoch: {run_epochs}, start global step: {run_steps}')

    if tensor_board_name is None:
        tensor_board_name = f'runs-{ds_config.interval}'

    date_str = date.today().isoformat()
    checkpoint_dir = f'{artifacts_dir}/{date_str}'
    tensor_board_runs_dir = f'{out_dir}/{tensor_board_name}/{model_dir}/{date_str}'

    optimizer, model_opt, \
        global_steps = train_and_log(model, model_config, ds_config,
                                     epochs=epochs,
                                     encoder=encoder,
                                     device=device,
                                     model_name=model_name,
                                     print_pred_detail=print_pred_detail,
                                     write_tensor_board=write_tensor_board,
                                     tensor_board_runs_dir=tensor_board_runs_dir,
                                     save_cp_every_epochs=save_cp_every_epochs,
                                     checkpoint_dir=checkpoint_dir,
                                     start_epoch=run_epochs,
                                     start_global_steps=run_steps,
                                     optimizer_state=optimizer_state,
                                     model_opt_state=model_opt_state,
                                     )

    save_path = f'{artifacts_dir}/{model_name}_final.pt'
    save_checkpoint(save_path, model, model_name, model_config, ds_config,
                    plan_epochs=epochs,
                    run_epochs=epochs,
                    run_steps=global_steps,
                    encoder=encoder,
                    optimizer_state=optimizer.state_dict(),
                    model_opt_state=model_opt.state_dict(),
                    )

    print(datetime.now())


if __name__ == "__main__":
    fire.Fire(main)
