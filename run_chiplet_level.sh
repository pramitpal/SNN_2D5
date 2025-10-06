#!/usr/bin/env bash

chiplet_id=$1
./booksim Chiplet_level_config > out_chiplet_${chiplet_id}.txt
echo "BookSim output saved to out_chiplet_${chiplet_id}.txt"
