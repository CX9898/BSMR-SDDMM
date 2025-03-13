#!/bin/bash

wget -nc https://storage.googleapis.com/sgk-sc2020/dlmc.tar.gz

tar -xvf dlmc.tar.gz

mv -n dlmc ../dataset/
mv -n ../dataset/2048_512_dlmc_data.txt dataset/dlmc/