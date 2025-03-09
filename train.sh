#python3 train_mnist.py --base "normal" --n_epochs 3
#python3 train_mnist.py --base "mppca" --model_file "model_c_50_l_5.pth" --n_epochs 3

#python3 train_cifar10.py --base "mppca"
#python3 train_celeba.py --base "mppca"
#python3 train_fgvc_aircraft.py --base "mppca"

#python3 train_cifar10.py --base "normal"
#python3 train_celeba.py --base "normal"
#python3 train_fgvc_aircraft.py --base "normal"

#python3 train_images.py --base "mppca" --flow "otcfm" --dataset "fgvc-aircraft" --data_dir "./data/fgvc-aircraft-2013b/data/images/"
#python3 train_images.py --base "normal" --flow "otcfm" --dataset "fgvc-aircraft" --data_dir "./data/fgvc-aircraft-2013b/data/images/"
#python3 train_images.py --base "mppca" --flow "cfm" --dataset "fgvc-aircraft" --data_dir "./data/fgvc-aircraft-2013b/data/images/"
#python3 train_images.py --base "normal" --flow "cfm" --dataset "fgvc-aircraft" --data_dir "./data/fgvc-aircraft-2013b/data/images/"

#python3 train_images.py --base "Normal" --flow "VPCFM" --dataset "fashion"
#python3 train_images.py --base "MPPCA" --flow "VPCFM" --dataset "fashion"
#python3 train_images.py --base "Normal" --flow "OTCFM" --dataset "fashion"
#python3 train_images.py --base "MPPCA" --flow "OTCFM" --dataset "fashion"

#python3 train_images.py --base "MPPCA" --flow "CFM" --dataset "celeba"
#python3 train_images.py --base "Normal" --flow "CFM" --dataset "celeba"
python3 train_images.py --base "MPPCA" --flow "OTCFM" --dataset "celeba"
python3 train_images.py --base "Normal" --flow "OTCFM" --dataset "celeba"

#python3 train_images.py --base "MPPCA" --flow "CFM" --dataset "fgvc-aircraft"
#python3 train_images.py --base "Normal" --flow "CFM" --dataset "fgvc-aircraft"
#python3 train_images.py --base "MPPCA" --flow "OTCFM" --dataset "fgvc-aircraft"
#python3 train_images.py --base "Normal" --flow "OTCFM" --dataset "fgvc-aircraft"

#python3 train_uci.py --flow "CFM" --base 'MPPCA' --dataset "POWER"
#python3 train_uci.py --flow "CFM" --base 'Normal' --dataset "POWER"
#python3 train_uci.py --flow "OTCFM" --base 'MPPCA' --dataset "POWER"
#python3 train_uci.py --flow "OTCFM" --base 'Normal' --dataset "POWER"

#python3 train_uci.py --flow "CFM" --base 'MPPCA' --dataset "GAS"
#python3 train_uci.py --flow "CFM" --base 'Normal' --dataset "GAS"
#python3 train_uci.py --flow "OTCFM" --base 'MPPCA' --dataset "GAS"
#python3 train_uci.py --flow "OTCFM" --base 'Normal' --dataset "GAS"

#python3 train_uci.py --flow "CFM" --base 'MPPCA' --dataset "HEPMASS"
#python3 train_uci.py --flow "CFM" --base 'Normal' --dataset "HEPMASS"
#python3 train_uci.py --flow "OTCFM" --base 'MPPCA' --dataset "HEPMASS"
#python3 train_uci.py --flow "OTCFM" --base 'Normal' --dataset "HEPMASS"

#python3 train_uci.py --flow "CFM" --base 'MPPCA' --dataset "MINIBOONE"
#python3 train_uci.py --flow "CFM" --base 'Normal' --dataset "MINIBOONE"
#python3 train_uci.py --flow "OTCFM" --base 'MPPCA' --dataset "MINIBOONE"
#python3 train_uci.py --flow "OTCFM" --base 'Normal' --dataset "MINIBOONE"

#python3 train_uci.py --flow "CFM" --base 'MPPCA' --dataset "BSDS300"
#python3 train_uci.py --flow "CFM" --base 'Normal' --dataset "BSDS300"
#python3 train_uci.py --flow "OTCFM" --base 'MPPCA' --dataset "BSDS300"
#python3 train_uci.py --flow "OTCFM" --base 'Normal' --dataset "BSDS300"
