#!/bin/bash
wget https://storage.googleapis.com/sgk-sc2020/dlmc.tar.gz

tar -xvf dlmc.tar.gz

mv dlmc dataset/
mv dataset/2048_512_dlmc_data.txt dataset/dlmc/