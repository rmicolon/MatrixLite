# MatrixLite
A lightweight library for matrices manipulations written in C++, using OpenMP to optimize the "embarrassingly parallel" parts of the code.

## Installation
Just copy `Matrix.h` and `Matrix.tpp` in your project directory and include the `Matrix.h` header.  
(Optionnal) If you want to make use of multi-threading you'll have to compile your project with OpenMP ([See here for supported compilers](https://www.openmp.org/resources/openmp-compilers-tools/)).  
E.g. using GCC on Linux: `g++ -fopenmp tutorial.cpp`

## Usage
Take a look at `tutorial.cpp` to see what you can do with the library.
