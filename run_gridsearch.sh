#!/usr/bin/env bash

lr_train=(0.01 0.001)
dropout_rnn=(0 0.1)
cnn=(dense globalpool)

for lr in ${lr_train[@]} ; do
    for dropout in ${dropout_rnn[@]} ; do
            for cnn_top in ${cnn[@]} ; do
            export lr dropout cnn_top
            sbatch runfile_grid.sh
            done
        done
done