# How to compile the cpp versions?

Use the following command in mac.

```
c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` aligator.cpp -o aligator`python3-config --extension-suffix`
```