//
// Created by DEV on 2022/1/13.
//

#ifndef CPPRACTICE_TEMPLATES_H
#define CPPRACTICE_TEMPLATES_H

#include <stdint.h>

template <typename T, int Size> //模板参数为整型数字
struct Array{
    T data[Size]
};

template <int i>
class A{
public:
    void foo(int){

    }
};

template <uint8_t a, typename b, void* c>
class B{};

template <bool, void (*a)()> class C{};

//其中的 (int) 为参数类型
template <void (A<3>::*a)(int)> class D{};

//template <float a> class E{};//error，只能是整型数

template <int i>
int Add(int a){
    return a + i;
}

void foo(){
    A<5> a;
    B<7, A<5>, nullptr> b;//模板参数可以是模板生成的类；也可以是一个指针
    C<false, &foo> c;
    D<&A<3>::foo> d;
    int x = Add<3>(5);//x == 8
}

#endif //CPPRACTICE_TEMPLATES_H
