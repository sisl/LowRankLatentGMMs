python3 train_uci.py --base "MPPCA"  --flow "OTCFM" --dataset "HEPMASS"
python3 train_uci.py --base "Normal"  --flow "OTCFM" --dataset "HEPMASS"
python3 train_uci.py --base "MPPCA"  --flow "VPCFM" --dataset "HEPMASS"
python3 train_uci.py --base "Normal"  --flow "VPCFM" --dataset "HEPMASS"

python3 train_uci.py --base "MPPCA"  --flow "OTCFM" --dataset "MINIBOONE"
python3 train_uci.py --base "Normal"  --flow "OTCFM" --dataset "MINIBOONE"
python3 train_uci.py --base "MPPCA"  --flow "VPCFM" --dataset "MINIBOONE"
python3 train_uci.py --base "Normal"  --flow "VPCFM" --dataset "MINIBOONE"

python3 train_uci.py --base "MPPCA"  --flow "OTCFM" --dataset "BSDS300"
python3 train_uci.py --base "Normal"  --flow "OTCFM" --dataset "BSDS300"
python3 train_uci.py --base "MPPCA"  --flow "VPCFM" --dataset "BSDS300"
python3 train_uci.py --base "Normal"  --flow "VPCFM" --dataset "BSDS300"

python3 train_images.py --base "MPPCA" --flow "OTCFM" --dataset "fashion" --epochs 100
python3 train_images.py --base "Normal" --flow "OTCFM" --dataset "fashion" --epochs 100
python3 train_images.py --base "MPPCA" --flow "VPCFM" --dataset "fashion" --epochs 100
python3 train_images.py --base "Normal" --flow "VPCFM" --dataset "fashion" --epochs 100

python3 train_images.py --base "MPPCA" --flow "OTCFM" --dataset "cifar10" --epochs 100
python3 train_images.py --base "Normal" --flow "OTCFM" --dataset "cifar10" --epochs 100
python3 train_images.py --base "MPPCA" --flow "VPCFM" --dataset "cifar10" --epochs 100
python3 train_images.py --base "Normal" --flow "VPCFM" --dataset "cifar10" --epochs 100

python3 train_images.py --base "MPPCA" --flow "OTCFM" --dataset "celeba" --epochs 50
python3 train_images.py --base "Normal" --flow "OTCFM" --dataset "celeba" --epochs 50
python3 train_images.py --base "MPPCA" --flow "VPCFM" --dataset "celeba" --epochs 50
python3 train_images.py --base "Normal" --flow "VPCFM" --dataset "celeba" --epochs 50

python3 train_images.py --base "MPPCA" --flow "VPCFM" --dataset "fashion" -epochs 100 --n_factors 3
python3 train_images.py --base "MPPCA" --flow "VPCFM" --dataset "fashion" -epochs 100 --n_factors 9
python3 train_images.py --base "MPPCA" --flow "VPCFM" --dataset "fashion" -epochs 100 --n_factors 12
python3 train_images.py --base "MPPCA" --flow "VPCFM" --dataset "fashion" -epochs 100 --n_factors 15
python3 train_images.py --base "MPPCA" --flow "VPCFM" --dataset "fashion" -epochs 100 --n_factors 18
