#python3 train_mnist.py --base "normal" --n_epochs 3
#python3 train_mnist.py --base "mppca" --model_file "model_c_50_l_5.pth" --n_epochs 3

#python3 train_cifar10.py --base "mppca"
#python3 train_celeba.py --base "mppca"
#python3 train_fgvc_aircraft.py --base "mppca"

#python3 train_cifar10.py --base "normal"
#python3 train_celeba.py --base "normal"
#python3 train_fgvc_aircraft.py --base "normal"

python3 uci_test.py --dataset "POWER"
python3 uci_test.py --dataset "GAS"
python3 uci_test.py --dataset "HEPMASS"
python3 uci_test.py --dataset "MINIBOONE"
python3 uci_test.py --dataset "BSDS300"
