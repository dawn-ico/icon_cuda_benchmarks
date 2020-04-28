#!/bin/bash

atlasInc=/home/giacomo/atlas-install/include
eckitInc=/home/giacomo/eckit-install/include

atlasLib=/home/giacomo/atlas-install/lib
eckitLib=/home/giacomo/eckit-install/lib

g++ -o out test_atlas_views.cpp -std=c++17 -I$atlasInc -I$eckitInc -L$atlasLib -latlas -L$eckitLib -leckit
