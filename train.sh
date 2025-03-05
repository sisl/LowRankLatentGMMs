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

#python3 train_images.py --base "mppca" --flow "cfm" --dataset "fashion"
#python3 train_images.py --base "normal" --flow "cfm" --dataset "fashion"
#python3 train_images.py --base "normal" --flow "otcfm" --dataset "fashion"
#python3 train_images.py --base "mppca" --flow "otcfm" --dataset "fashion"

#python3 train_images.py --base "normal" --flow "otcfm" --dataset "fgvc-aircraft"
#python3 train_images.py --base "mppca" --flow "otcfm" --dataset "fgvc-aircraft"


#python3 train_uci.py --flow "cfm" --base 'mppca' --dataset "power"
#python3 train_uci.py --flow "cfm" --base 'normal' --dataset "power"
#python3 train_uci.py --flow "otcfm" --base 'mppca' --dataset "power"
#python3 train_uci.py --flow "otcfm" --base 'normal' --dataset "power"

#python3 train_uci.py --flow "cfm" --base 'mppca' --dataset "gas"
#python3 train_uci.py --flow "cfm" --base 'normal' --dataset "gas"
#python3 train_uci.py --flow "otcfm" --base 'mppca' --dataset "gas"
#python3 train_uci.py --flow "otcfm" --base 'normal' --dataset "gas"

#python3 train_uci.py --flow "CFM" --base 'MPPCA' --dataset "HEPMASS"
python3 train_uci.py --flow "CFM" --base 'Normal' --dataset "HEPMASS"
#python3 train_uci.py --flow "otcfm" --base 'mppca' --dataset "hepmass"
#python3 train_uci.py --flow "otcfm" --base 'normal' --dataset "hepmass"

#python3 train_uci.py --flow "cfm" --base 'mppca' --dataset "miniboone"
#python3 train_uci.py --flow "cfm" --base 'normal' --dataset "miniboone"
#python3 train_uci.py --flow "otcfm" --base 'mppca' --dataset "miniboone"
#python3 train_uci.py --flow "otcfm" --base 'normal' --dataset "miniboone"

#python3 train_uci.py --flow "cfm" --base 'mppca' --dataset "bsds300"
#python3 train_uci.py --flow "cfm" --base 'normal' --dataset "bsds300"
#python3 train_uci.py --flow "otcfm" --base 'mppca' --dataset "bsds300"
#python3 train_uci.py --flow "otcfm" --base 'normal' --dataset "bsds300"
