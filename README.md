### tips
不用在cpp文件里去直接include 后缀为.cu的代码，容易编译不同过。
只需在cpp文件里声明要用的函数，在cu里定义这个函数，然后将所有的cu文件生成一个动态库就可以了

### compile
```
mkdir build
cd build
cmake ..
make
```
