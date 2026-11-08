cmake_minimum_required(VERSION 3.22.1)
project(orchnn C)

add_library(orchnn SHARED kernels.c)
target_compile_options(orchnn PRIVATE -O3 -ffast-math -fopenmp)
target_link_options(orchnn PRIVATE -fopenmp -static-openmp -Wl,-z,max-page-size=16384)
