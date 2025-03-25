#! /bin/bash

python cifar10.py --feedforward_linear_layer bitlinear --attention_linear_layer bitlinear

python cifar10.py --feedforward_linear_layer trilinear --attention_linear_layer trilinear

python cifar10.py --feedforward_linear_layer tliner --attention_linear_layer tlinear
python cifar10.py --feedforward_linear_layer tliner_channel --attention_linear_layer tlinear_channel
python cifar10.py --feedforward_linear_layer tliner_group --attention_linear_layer tlinear_group

python cifar10.py --feedforward_linear_layer linear --attention_linear_layer linear