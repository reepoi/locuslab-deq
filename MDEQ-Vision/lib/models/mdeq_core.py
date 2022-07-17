from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import functools
from termcolor import colored

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._utils
import torch.nn.functional as F
import torch.autograd as autograd

sys.path.append("lib/")
from utils.utils import get_world_size, get_rank

sys.path.append("../")
from lib.optimizations import VariationalHidDropout2d, weight_norm
from lib.solvers import anderson, broyden
from lib.jacobian import jac_loss_estimate, power_method
from lib.layer_utils import list2vec, vec2list, norm_diff, conv3x3, conv5x5


BN_MOMENTUM = 0.1
BLOCK_GN_AFFINE = True    # Don't change the value here. The value is controlled by the yaml files.
FUSE_GN_AFFINE = True     # Don't change the value here. The value is controlled by the yaml files.
POST_GN_AFFINE = True     # Don't change the value here. The value is controlled by the yaml files.
DEQ_EXPAND = 5        # Don't change the value here. The value is controlled by the yaml files.
NUM_GROUPS = 4        # Don't change the value here. The value is controlled by the yaml files.
logger = logging.getLogger(__name__)


class BasicBlock(nn.Module):
    """
    A canonical residual block with two 3x3 convolutions and an intermediate ReLU. Corresponds to Figure 2
    in the paper.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, n_big_kernels=0, dropout=0.0, wnorm=False):
        super(BasicBlock, self).__init__()
        conv1 = conv5x5 if n_big_kernels >= 1 else conv3x3
        conv2 = conv5x5 if n_big_kernels >= 2 else conv3x3
        inner_planes = int(DEQ_EXPAND*planes)

        self.conv1 = conv1(inplanes, inner_planes)
        self.gn1 = nn.GroupNorm(NUM_GROUPS, inner_planes, affine=BLOCK_GN_AFFINE)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = conv2(inner_planes, planes)
        self.gn2 = nn.GroupNorm(NUM_GROUPS, planes, affine=BLOCK_GN_AFFINE)

        self.gn3 = nn.GroupNorm(NUM_GROUPS, planes, affine=BLOCK_GN_AFFINE)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.drop = VariationalHidDropout2d(dropout)
        if wnorm: self._wnorm()
    
    def _wnorm(self):
        """
        Register weight normalization
        """
        self.conv1, self.conv1_fn = weight_norm(self.conv1, names=['weight'], dim=0)
        self.conv2, self.conv2_fn = weight_norm(self.conv2, names=['weight'], dim=0)
    
    def _reset(self, bsz, d, H, W):
        """
        Reset dropout mask and recompute weight via weight normalization
        """
        if 'conv1_fn' in self.__dict__:
            self.conv1_fn.reset(self.conv1)
        if 'conv2_fn' in self.__dict__:
            self.conv2_fn.reset(self.conv2)
        self.drop.reset_mask(bsz, d, H, W)
            
    def forward(self, x, injection=None):
        if injection is None: injection = 0
        residual = x

        out = self.relu(self.gn1(self.conv1(x)))
        out = self.drop(self.conv2(out)) + injection
        out = self.gn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.gn3(self.relu3(out))
        return out
    
       
blocks_dict = { 'BASIC': BasicBlock }


class BranchNet(nn.Module):
    """
    The residual block part of each resolution stream in Figure 1.
    Note Figure 1's residual block part shows only one residual block.
    The residual block used is the class BasicBlock.
    """

    def __init__(self, blocks): # blocks: list[BasicBlock]
        super().__init__()
        self.blocks = blocks
    
    def forward(self, x, injection=None):
        """
        Feeds x through all of the residual blocks in the residual block part
        of each resolution stream in Figure 1.
        Note all the provided experiment config files use one residual block.
        (len(self.blocks) = 1).
        """
        y = self.blocks[0](x, injection)
        for block in self.blocks[1:]:
            y = block(y)
        return y
    
    
class DownsampleModule(nn.Module):
    def __init__(self, num_channels, in_res, out_res):
        """
        A downsample step from resolution j (with in_res) to resolution i (with out_res). A series of 2-strided convolutions.
        """
        super(DownsampleModule, self).__init__()
        # downsample (in_res=j, out_res=i)
        convs = []
        inp_chan = num_channels[in_res]
        out_chan = num_channels[out_res]
        level_diff = out_res - in_res
        
        kwargs = {"kernel_size": 3, "stride": 2, "padding": 1, "bias": False}
        for k in range(level_diff):
            intermediate_out = out_chan if k == (level_diff-1) else inp_chan
            components = [('conv', nn.Conv2d(inp_chan, intermediate_out, **kwargs)), 
                          ('gnorm', nn.GroupNorm(NUM_GROUPS, intermediate_out, affine=FUSE_GN_AFFINE))]
            if k != (level_diff-1):
                components.append(('relu', nn.ReLU(inplace=True)))
            convs.append(nn.Sequential(OrderedDict(components)))
        self.net = nn.Sequential(*convs)  
            
    def forward(self, x):
        return self.net(x)


class UpsampleModule(nn.Module):
    def __init__(self, num_channels, in_res, out_res):
        """
        An upsample step from resolution j (with in_res) to resolution i (with out_res). 
        Simply a 1x1 convolution followed by an interpolation.
        """
        super(UpsampleModule, self).__init__()
        # upsample (in_res=j, out_res=i)
        inp_chan = num_channels[in_res]
        out_chan = num_channels[out_res]
        level_diff = in_res - out_res
        
        self.net = nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(inp_chan, out_chan, kernel_size=1, bias=False)),
                        ('gnorm', nn.GroupNorm(NUM_GROUPS, out_chan, affine=FUSE_GN_AFFINE)),
                        ('upsample', nn.Upsample(scale_factor=2**level_diff, mode='nearest'))]))
        
    def forward(self, x):
        return self.net(x)

    
class MDEQModule(nn.Module):
    def __init__(self, num_branches, block_class, num_blocks, num_channels, big_kernels, dropout=0.0):
        """
        An MDEQ layer (note that MDEQ only has one layer). 
        This runs the transformations in the large tan box in Figure 1:
        residual blocks, and multi-resolution fusion.
        """
        super(MDEQModule, self).__init__()

        self.num_branches = num_branches
        self.num_channels = num_channels
        self.big_kernels = big_kernels

        self.branches = self._make_branches(num_branches, block_class, num_blocks, num_channels, big_kernels, dropout=dropout)
        self.fuse_layers = self._make_fuse_layers()
        self.post_fuse_layers = nn.ModuleList()
        for nc in num_channels:
            pfl = nn.Sequential(OrderedDict([
                    ('relu', nn.ReLU(False)),
                    ('conv', nn.Conv2d(nc, nc, kernel_size=1, bias=False)),
                    ('gnorm', nn.GroupNorm(NUM_GROUPS // 2, nc, affine=POST_GN_AFFINE))
            ]))
            self.post_fuse_layers.append(pfl)
    
    def _wnorm(self):
        """
        Apply weight normalization to the learnable parameters of MDEQ
        """
        self.post_fuse_fns = []
        for branch, pfl in zip(self.branches, self.post_fuse_layers, strict=True):
            for block in branch.blocks:
                block._wnorm()
            conv, fn = weight_norm(pfl.conv, names=['weight'], dim=0)
            self.post_fuse_fns.append(fn)
            pfl.conv = conv
        
        # Throw away garbage
        torch.cuda.empty_cache()
        
    def _reset(self, xs):
        """
        Reset the dropout mask and the learnable parameters (if weight normalization is applied)
        """
        for i, branch in enumerate(self.branches):
            for block in branch.blocks:
                block._reset(*xs[i].shape)
            if self.wnorm:
                self.post_fuse_fns[i].reset(self.post_fuse_layers[i].conv)    # Re-compute (...).conv.weight using _g and _v

    def _make_branches(self, num_branches, block_class, num_blocks, num_channels, big_kernels, dropout=0.0):
        """
        Make the residual block (s; default=1 block) of MDEQ's f_\theta layer. Specifically,
        it returns `branch_layers[i]` gives the module that operates on input from resolution i.
        Each branch contains `num_blocks` of resdiual blocks of type `block_class`.
        """
        branches = nn.ModuleList()
        for nc, nbk, nb in zip(num_channels, big_kernels, num_blocks, strict=True):
            nb_blocks = [block_class(nc, nc, n_big_kernels=nbk, dropout=dropout) for _ in range(nb)]
            branches.append(BranchNet(nb_blocks))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        """
        Create the multiscale fusion layer (which does simultaneous up- and downsamplings).
        Multi-resolution Fusion part of Figure 1.
        """
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_channels = self.num_channels
        fuse_layers = []
        for i in range(num_branches):
            fuse_layer = []                    # The fuse modules into branch #i
            for j in range(num_branches):
                if i == j:
                    fuse_layer.append(None)    # Identity if the same branch
                else:
                    module = UpsampleModule if j > i else DownsampleModule
                    fuse_layer.append(module(num_channels, in_res=j, out_res=i))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        # fuse_layers[i][j] gives the (series of) conv3x3s that convert input from branch j to branch i
        return nn.ModuleList(fuse_layers)

    def forward(self, x, injection, *args):
        """
        The two steps of a multiscale DEQ module (see paper): a per-resolution residual block and 
        a parallel multiscale fusion step.
        """
        if injection is None:
            injection = torch.zeros(len(x)) # len(x) = x.size(dim=0)

        # Step 1: Per-resolution residual block
        x_block = [branch(x[i], injection[i]) for i, branch in enumerate(self.branches)]
        if len(x_block) == 1:
            return x_block

        # Step 2: Multiscale fusion
        x_fuse = []
        for i, fuse_layer in enumerate(self.fuse_layers):
            # sampleModule is either a DownsampleModule or an UpsampleModule
            y = sum(x_block[j] if i == j else sampleModule(x_block[j]) for j, sampleModule in enumerate(fuse_layer))
            x_fuse.append(self.post_fuse_layers[i](y))

        return x_fuse


class MDEQNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        """
        Build an MDEQ model with the given hyperparameters

        Args:
            cfg ([config]): The configuration file (parsed from yaml) specifying the model settings
        """
        super(MDEQNet, self).__init__()
        global BN_MOMENTUM
        BN_MOMENTUM = kwargs.get('BN_MOMENTUM', 0.1)
        self.parse_cfg(cfg)
        init_chansize = self.init_chansize

        self.downsample = nn.Sequential(
            conv3x3(3, init_chansize, stride=(2 if self.downsample_times >= 1 else 1)),
            nn.BatchNorm2d(init_chansize, momentum=BN_MOMENTUM, affine=True),
            nn.ReLU(inplace=True),
            conv3x3(init_chansize, init_chansize, stride=(2 if self.downsample_times >= 2 else 1)),
            nn.BatchNorm2d(init_chansize, momentum=BN_MOMENTUM, affine=True),
            nn.ReLU(inplace=True))

        if self.downsample_times > 2:
            for i in range(3, self.downsample_times+1):
                self.downsample.add_module(f"DS{i}", conv3x3(init_chansize, init_chansize, stride=2))
                self.downsample.add_module(f"DS{i}-BN", nn.BatchNorm2d(init_chansize, momentum=BN_MOMENTUM, affine=True))
                self.downsample.add_module(f"DS{i}-RELU", nn.ReLU(inplace=True))
        
        # PART I: Input injection module
        if self.downsample_times == 0 and self.num_branches <= 2:
            # We use the downsample module above as the injection transformation
            self.stage0 = None
        else:
            self.stage0 = nn.Sequential(nn.Conv2d(self.init_chansize, self.init_chansize, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(self.init_chansize, momentum=BN_MOMENTUM, affine=True),
                                        nn.ReLU(inplace=True))
        
        # PART II: MDEQ's f_\theta layer
        self.fullstage = self._make_stage(self.fullstage_cfg, self.num_channels, dropout=self.dropout)
        self.alternative_mode = "abs" if self.stop_mode == "rel" else "rel"
        if self.wnorm:
            self.fullstage._wnorm()

        self.iodrop = VariationalHidDropout2d(0.0)
        self.hook = None
        
    def parse_cfg(self, cfg):
        """
        Parse a configuration file
        """
        global DEQ_EXPAND, NUM_GROUPS, BLOCK_GN_AFFINE, FUSE_GN_AFFINE, POST_GN_AFFINE
        self.num_branches = cfg['MODEL']['EXTRA']['FULL_STAGE']['NUM_BRANCHES']
        self.num_channels = cfg['MODEL']['EXTRA']['FULL_STAGE']['NUM_CHANNELS']
        self.init_chansize = self.num_channels[0]
        self.num_layers = cfg['MODEL']['NUM_LAYERS']
        self.dropout = cfg['MODEL']['DROPOUT']
        self.wnorm = cfg['MODEL']['WNORM']
        self.num_classes = cfg['MODEL']['NUM_CLASSES']
        self.downsample_times = cfg['MODEL']['DOWNSAMPLE_TIMES']
        self.fullstage_cfg = cfg['MODEL']['EXTRA']['FULL_STAGE']   
        self.pretrain_steps = cfg['TRAIN']['PRETRAIN_STEPS']

        # DEQ related
        self.f_solver = eval(cfg['DEQ']['F_SOLVER'])
        self.b_solver = eval(cfg['DEQ']['B_SOLVER'])
        if self.b_solver is None:
            self.b_solver = self.f_solver
        self.f_thres = cfg['DEQ']['F_THRES']
        self.b_thres = cfg['DEQ']['B_THRES']
        self.stop_mode = cfg['DEQ']['STOP_MODE']
        
        # Update global variables
        DEQ_EXPAND = cfg['MODEL']['EXPANSION_FACTOR']
        NUM_GROUPS = cfg['MODEL']['NUM_GROUPS']
        BLOCK_GN_AFFINE = cfg['MODEL']['BLOCK_GN_AFFINE']
        FUSE_GN_AFFINE = cfg['MODEL']['FUSE_GN_AFFINE']
        POST_GN_AFFINE = cfg['MODEL']['POST_GN_AFFINE']

        self._validate_cfg()

    def _validate_cfg(self):
        num_branches = self.fullstage_cfg['NUM_BRANCHES']
        num_blocks = self.fullstage_cfg['NUM_BLOCKS']
        block_class = self.fullstage_cfg[layer_config['BLOCK']]
        big_kernels = self.fullstage_cfg['BIG_KERNELS']

        error_msg = ''
        if self.num_branches != len(num_blocks):
            error_msg = 'Must have len(NUM_BLOCKS) equal NUM_BRANCHES'
        elif num_branches != len(self.num_channels):
            error_msg = 'Must have len(NUM_CHANNELS) equal NUM_BRANCHES'
        elif num_branches != len(big_kernels):
            error_msg = 'Must have len(BIG_KERNELS) equal NUM_BRANCHES'
        if error_msg:
            logger.error(error_msg)
            raise ValueError(error_msg)
            
    def _make_stage(self, layer_config, num_channels, dropout=0.0):
        """
        Build an MDEQ block with the given hyperparameters
        """
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        block_class = blocks_dict[layer_config['BLOCK']]
        big_kernels = layer_config['BIG_KERNELS']
        return MDEQModule(num_branches, block_class, num_blocks, num_channels,
                          big_kernels, dropout=dropout)

    def _forward(self, x, train_step=-1, compute_jac_loss=True, spectral_radius_mode=False, writer=None, **kwargs):
        """
        The core MDEQ module. In the starting phase, we can (optionally) enter
        a shallow stacked f_\theta training mode to warm up the weights
        (specified by the self.pretrain_steps; see below)
        """
        num_branches = self.num_branches
        f_thres = kwargs.get('f_thres', self.f_thres)
        b_thres = kwargs.get('b_thres', self.b_thres)
        x = self.downsample(x)
        rank = get_rank()
        
        # Inject only to the highest resolution.
        # The remaining injections of x_list are zero.
        x_list = [self.stage0(x) if self.stage0 else x]
        bsz, _, H, W = x_list[-1].shape
        for channel_count in self.num_channels[1:]:
            bsz, _, H, W = x_list[-1].shape
            x_list.append(torch.zeros(bsz, channel_count, H//2, W//2).to(x))
            
        z_list = [torch.zeros_like(elem) for elem in x_list]
        z1 = list2vec(z_list)
        cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in z_list]
        func = lambda z: list2vec(self.fullstage(vec2list(z, cutoffs), x_list))
        
        # For variational dropout mask resetting and weight normalization re-computations
        self.fullstage._reset(z_list)

        jac_loss = torch.tensor(0.0).to(x)
        sradius = torch.zeros(bsz, 1).to(x)
        deq_mode = (train_step < 0) or (train_step >= self.pretrain_steps)
        
        # Multiscale Deep Equilibrium!
        if not deq_mode:
            for _ in range(self.num_layers):
                z1 = func(z1)
            new_z1 = z1

            if self.training:
                if compute_jac_loss:
                    z2 = z1.clone().detach().requires_grad_()
                    new_z2 = func(z2)
                    jac_loss = jac_loss_estimate(new_z2, z2)
        else:
            with torch.no_grad():
                result = self.f_solver(func, z1, threshold=f_thres, stop_mode=self.stop_mode, name="forward")
                z1 = result['result']
            new_z1 = z1

            if (not self.training) and spectral_radius_mode:
                with torch.enable_grad():
                    new_z1 = func(z1.requires_grad_())
                _, sradius = power_method(new_z1, z1, n_iters=150)

            if self.training:
                new_z1 = func(z1.requires_grad_())
                if compute_jac_loss:
                    jac_loss = jac_loss_estimate(new_z1, z1)
                    
                def backward_hook(grad):
                    if self.hook is not None:
                        self.hook.remove()
                        torch.cuda.synchronize()
                    result = self.b_solver(lambda y: autograd.grad(new_z1, z1, y, retain_graph=True)[0] + grad, torch.zeros_like(grad), 
                                          threshold=b_thres, stop_mode=self.stop_mode, name="backward")
                    return result['result']
                self.hook = new_z1.register_hook(backward_hook)
                
        y_list = self.iodrop(vec2list(new_z1, cutoffs))
        return y_list, jac_loss.view(1,-1), sradius.view(-1,1)
    
    def forward(self, x, train_step=-1, **kwargs):
        raise NotImplemented    # To be inherited & implemented by MDEQClsNet and MDEQSegNet (see mdeq.py)
