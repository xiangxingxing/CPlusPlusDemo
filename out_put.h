//
// Created by DEV on 2021/10/27.
//

#ifndef CPPRACTICE_OUT_PUT_H
#define CPPRACTICE_OUT_PUT_H


class out_put {
public:
    static void BasicUserInputDemo();
    static void BasicOutputFormatDemo();
    static void StringDemo();
    static void StringOperationDemo();

public:
    template<typename T> void Swap(T &a, T &b);

private:
    static int m_total;
};

template<typename T>
void out_put::Swap(T &a, T &b) {
    T temp = a;
    a = b;
    b = temp;
}

#endif //CPPRACTICE_OUT_PUT_H
