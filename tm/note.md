```
nohup tensorboard --logdir=out/runs-4h --port 8094 --path_prefix=/4h --bind_all > out/logs/tensorboard.nohup &

runs-15m 8092
runs-1h 8093
runs-4h 8094
runs-1d 8095

```

```
cd tm

mkdir out/logs/`date +%F`



PYTHONPATH=.. nohup python train.py --symbol=ETHUSDT --kl_interval=15m --epochs=20 > out/logs/`date +%F`/mdo-32-eth-15m-`date +%H_%M`.nohup &

PYTHONPATH=.. nohup python train.py --symbol=btcusdt --kl_interval=4h --epochs=20 --seq_len=16 --model_type=tred > out/logs/`date +%F`/med-`date +%H_%M`.nohup &



```

```
cd tm

PYTHONPATH=.. nohup python train_cont.py --epochs=40 --cp_file=out/artifacts/mdo/..._final.pt  > out/logs/`date +%F`/mdo-sl16-b6-btc-1h-cont-`date +%H_%M`.nohup &

```

```
cd tm

PYTHONPATH=.. python load_eval.py\
 --cp_file=out/artifacts/med/2023-07-01/ethusdt-4h-sl16-di34-dm1024-h8-b6x2-df1024-dg16_epoch-32.pt\
  --print_pred_detail --model_type=tred\
  --cached_batch_base_dir="../data/tm/batch_tensor_val_m3_9"
  --eval_batch --save_graph --export_onnx --show_state_dict\

PYTHONPATH=.. python load_eval.py --cp_file=out/artifacts/mdo/ethusdt-4h-sl32-di34-dm1024-h8-b12-df1024-dg16_final.pt --model_type=trd -- --interactive
PYTHONPATH=.. python load_eval.py -- --help


evaluate:

PYTHONPATH=.. python load_eval.py --cp_file=out/artifacts/med/btc-1h-sl32-di44-dm1024-h8-b3x2-df1024-dg16_final.pt \
 --evaluate_epoch --evaluate_steps=2

```

```
cd tm

PYTHONPATH=.. python ds.py --action=build_tensors

PYTHONPATH=.. python ds.py --action=check_tensor_batch

PYTHONPATH=.. python ds.py --action=check_csv_batch

```

