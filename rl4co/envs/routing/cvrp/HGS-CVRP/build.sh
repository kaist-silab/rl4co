#!/usr/bin/env bash

PWD=$(pwd)

cd "$(dirname "$0")"
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles"
make lib
cd $PWD
