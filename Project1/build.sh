cd lib_so
mkdir build
cd build
cmake ..
make -j4
echo "finish lib and os ..."

cd ../..
mkdir build
cd build
cmake ..
make -j4
cd ..
echo "run ./hello ..."