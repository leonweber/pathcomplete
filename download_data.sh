#!/bin/bash

mkdir -p data

#wget https://download.microsoft.com/download/0/E/4/0E4A272F-FB27-4DEE-BD3A-22C9116FC551/Release2.0.zip
wget http://www.pathwaycommons.org/archives/PC2/v10/PathwayCommons10.All.hgnc.sif.gz && gunzip PathwayCommons10.All.hgnc.sif.gz && mv PathwayCommons10.All.hgnc.sif data/

