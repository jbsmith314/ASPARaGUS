# Clean training
python3 brew_poison.py --poisonkey 0 --modelkey 0 --deterministic --optimization basic --only_clean_training --epochs 40                                                                                   --modelsave_path models/higher_budget/all_clean      --save_epoch 39 --targets 1000 --recipe patch                 --save_weights timed --budget 0.1

# First quarter poisoned
python3 brew_poison.py --poisonkey 0 --modelkey 0 --deterministic --optimization basic --skip_clean_training --epochs 10                                                                                   --modelsave_path models/higher_budget/first_quarter  --save_epoch 9  --targets 1000 --recipe patch                 --save_weights timed --budget 0.1
python3 brew_poison.py --poisonkey 0 --modelkey 0 --deterministic --optimization basic --only_clean_training --epochs 30 --load_model models/higher_budget/first_quarter/poisoned_model/full_epoch_9.pth   --modelsave_path models/higher_budget/first_quarter  --save_epoch 39 --targets 1000 --recipe patch --start_from 10 --save_weights timed --budget 0.1

# Second quarter poisoned
python3 brew_poison.py --poisonkey 0 --modelkey 0 --deterministic --optimization basic --only_clean_training --epochs 10                                                                                   --modelsave_path models/higher_budget/second_quarter --save_epoch 9  --targets 1000 --recipe patch                 --save_weights timed --budget 0.1
python3 brew_poison.py --poisonkey 0 --modelkey 0 --deterministic --optimization basic --skip_clean_training --epochs 10 --load_model models/higher_budget/second_quarter/clean_model/full_epoch_9.pth     --modelsave_path models/higher_budget/second_quarter --save_epoch 19 --targets 1000 --recipe patch --start_from 10 --save_weights timed --budget 0.1
python3 brew_poison.py --poisonkey 0 --modelkey 0 --deterministic --optimization basic --only_clean_training --epochs 20 --load_model models/higher_budget/second_quarter/poisoned_model/full_epoch_19.pth --modelsave_path models/higher_budget/second_quarter --save_epoch 39 --targets 1000 --recipe patch --start_from 20 --save_weights timed --budget 0.1

# Third quarter poisoned
python3 brew_poison.py --poisonkey 0 --modelkey 0 --deterministic --optimization basic --only_clean_training --epochs 20                                                                                   --modelsave_path models/higher_budget/third_quarter  --save_epoch 19 --targets 1000 --recipe patch                 --save_weights timed --budget 0.1
python3 brew_poison.py --poisonkey 0 --modelkey 0 --deterministic --optimization basic --skip_clean_training --epochs 10 --load_model models/higher_budget/third_quarter/clean_model/full_epoch_19.pth     --modelsave_path models/higher_budget/third_quarter  --save_epoch 29 --targets 1000 --recipe patch --start_from 20 --save_weights timed --budget 0.1
python3 brew_poison.py --poisonkey 0 --modelkey 0 --deterministic --optimization basic --only_clean_training --epochs 10 --load_model models/higher_budget/third_quarter/poisoned_model/full_epoch_29.pth  --modelsave_path models/higher_budget/third_quarter  --save_epoch 39 --targets 1000 --recipe patch --start_from 30 --save_weights timed --budget 0.1

# Fourth quarter poisoned
python3 brew_poison.py --poisonkey 0 --modelkey 0 --deterministic --optimization basic --only_clean_training --epochs 30                                                                                   --modelsave_path models/higher_budget/fourth_quarter --save_epoch 29 --targets 1000 --recipe patch                 --save_weights timed --budget 0.1
python3 brew_poison.py --poisonkey 0 --modelkey 0 --deterministic --optimization basic --skip_clean_training --epochs 10 --load_model models/higher_budget/fourth_quarter/clean_model/full_epoch_29.pth    --modelsave_path models/higher_budget/fourth_quarter --save_epoch 39 --targets 1000 --recipe patch --start_from 30 --save_weights timed --budget 0.1

# All poisoned
python3 brew_poison.py --poisonkey 0 --modelkey 0 --deterministic --optimization basic --skip_clean_training --epochs 40                                                                                   --modelsave_path models/higher_budget/all_poisoned   --save_epoch 39 --targets 1000 --recipe patch                 --save_weights timed --budget 0.1
