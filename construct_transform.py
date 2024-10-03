'''
Author: 
    Liam Kruse
Email: 
    lkruse@stanford.edu
Description:
    Functionality to create a normalizing flow transformation. Construct
    a composite transformation by chaining together linear transforms with 
    base transforms (such as neural spline flows and unconstrained monotonic
    neural networks)
'''

import torch

from nflows.nn.nets.resnet import ResidualNet
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform
)
from nflows.transforms.coupling import (
    AffineCouplingTransform,
    PiecewiseRationalQuadraticCouplingTransform,
)

from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform
)

from nflows.transforms.base import CompositeTransform
from nflows.transforms import LULinear
from nflows.transforms.permutations import ReversePermutation

# create a binary mask, credit nflows
def create_alternating_binary_mask(features, even=True):
    mask = torch.zeros(features).byte()
    start = 0 if even else 1
    mask[start::2] += 1
    return mask

# create linear transform, credit NSF
def create_linear_transform(args):
    if args.linear == 'permutation':
        return [ReversePermutation(features=args.features)]
    elif args.linear == 'lu':
        return [
            ReversePermutation(features=args.features),
            LULinear(args.features, identity_init=True)
        ]
    else:
        raise ValueError
    
# create base transform, credit NSF
def create_base_transform(i, args):
    if args.base == 'rq-ar':
        return [MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=args.features,
            hidden_features=args.hidden_features,
            context_features=args.context_features,
            num_bins=args.num_bins,
            tails='linear',
            tail_bound=args.tail_bound,
            use_residual_blocks=True,
            use_batch_norm=args.use_batch_norm
        )]
    elif args.base == 'rq-c':
        return [PiecewiseRationalQuadraticCouplingTransform(
            mask=create_alternating_binary_mask(args.features, even=(i%2==0)),
            transform_net_create_fn=lambda in_features, out_features: ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=args.hidden_features,
                context_features=args.context_features,
                use_batch_norm=args.use_batch_norm
            ),
            num_bins=args.num_bins,
            tails='linear',
            tail_bound=args.tail_bound,
        )]
    elif args.base == 'affine-ar':
        return [MaskedAffineAutoregressiveTransform(
            features=args.features,
            hidden_features=args.hidden_features,
            context_features=args.context_features,
            use_batch_norm=args.use_batch_norm,
            use_residual_blocks=True,
        )]
    elif args.base == 'affine-c':
        return [AffineCouplingTransform(
            mask=create_alternating_binary_mask(args.features, even=(i%2==0)),
            transform_net_create_fn=lambda in_features, out_features: ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=args.hidden_features,
                context_features=args.context_features,
                use_batch_norm=args.use_batch_norm
            )
        )]
    else:
        raise ValueError
    
# create flow transform
def create_transform(args):
    transform_list = []
    for i in range(args.num_flow_steps):
        transform_list.extend(create_linear_transform(args))
        transform_list.extend(create_base_transform(i, args))

    transform_list.extend(create_linear_transform(args))
    transform = CompositeTransform(transform_list)
    return transform
