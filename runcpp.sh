#!/usr/bin/bash

g++ -std=c++20 -Wall $1 && ./a.out > out && cat out
rm a.out
rm out
