from datetime import datetime, date

import fire

from tm.model.model_config import ModelConfig
from tm.model.trainer import train_and_log, gen_model_name, save_checkpoint
from tm.dataset.dataset_config import DatasetConfig, extra_intervals_map, get_d_input_by_interval
from tm.model_decoder_only.transformer import make_model as make_decoder_only_model
from tm.model_encoder_decoder.transformer import make_model as make_encoder_decoder_model
from tm.common.utils import set_print_options, check_device


def main(epochs: int = 10,
         steps: int = None,
         symbol='ETHUSDT',
         kl_interval='1d',
         model_type='encoder_decoder',  # decoder_only/encoder_decoder
         total_blocks=12,
         seq_len=32,
         batch_size=10,
         train_batches_per_step=1200,
         val_batches_per_step=100,
         print_pred_detail=True,
         write_tensor_board=True,
         tensor_board_name=None,
         market_data_base_dir='../data/tm/market',
         cached_batch_base_dir='../data/tm/batch_tensor',
         ds_split_method_name=None,
         use_cached_batch=True,
         save_cp_every_epochs=4,
         model_name_postfix=None
         ):
    print(datetime.now())

    set_print_options()

    device = check_device()

    dsc = DatasetConfig(
        base_dir=f'{market_data_base_dir}/kline-basic',
        symbol=symbol.upper(),
        interval=kl_interval,
        extra_intervals=extra_intervals_map.get(kl_interval),
        shuffle_buffer=200,
        seq_len=seq_len,
        batch_size=batch_size,
        train_batches_per_step=train_batches_per_step,
        val_batches_per_step=val_batches_per_step,
        cached_batch_base_dir=cached_batch_base_dir,
        use_cached_batch=use_cached_batch,
        ds_split_method_name=ds_split_method_name,
    )

    if model_type == 'decoder_only':
        blocks = total_blocks
        encoder = False
        model_dir = 'mdo'
    else:
        blocks = total_blocks // 2
        encoder = True
        model_dir = 'med'

    d_input = get_d_input_by_interval(kl_interval)
    mc = ModelConfig(
        d_input=d_input,
        blocks=blocks,
    )

    out_dir = 'out'

    artifacts_dir = f'{out_dir}/artifacts/{model_dir}'
    cp_base_dir = artifacts_dir

    if model_type == 'decoder_only':
        model = make_decoder_only_model(mc, device=device)
    else:  # encoder_decoder
        model = make_encoder_decoder_model(mc, device=device)

    model_name = gen_model_name(mc, dsc, encoder=encoder, postfix=model_name_postfix)
    print(f'model name: {model_name}')
    print(f'batch size: {batch_size}, batches per step: {train_batches_per_step}/{val_batches_per_step}')

    date_str = date.today().isoformat()
    checkpoint_dir = f'{cp_base_dir}/{date_str}'
    if tensor_board_name is None:
        tensor_board_name = f'runs-{kl_interval}'

    tensor_board_runs_dir = f'{out_dir}/{tensor_board_name}/{model_dir}/{date_str}'

    optimizer, model_opt, \
        global_steps = train_and_log(model, mc, dsc,
                                     epochs=epochs,
                                     steps=steps,
                                     encoder=encoder,
                                     device=device,
                                     model_name=model_name,
                                     print_pred_detail=print_pred_detail,
                                     write_tensor_board=write_tensor_board,
                                     tensor_board_runs_dir=tensor_board_runs_dir,
                                     save_cp_every_epochs=save_cp_every_epochs,
                                     checkpoint_dir=checkpoint_dir,
                                     )

    save_path = f'{artifacts_dir}/{model_name}_final.pt'
    save_checkpoint(save_path, model, model_name, mc, dsc,
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
