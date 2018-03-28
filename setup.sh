#!/bin/bash
set -e
echo "Setup project"
mkdir build
cd build/
cmake ..
make -j4
echo "Done with setup, ready to run, call:"
echo "./mainaimGraph configuration.xml"
echo "For more information please read README.markdown"