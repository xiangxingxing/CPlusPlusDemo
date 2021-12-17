//
// Created by DEV on 2021/10/28.
//

#ifndef CPPRACTICE_LIST_H
#define CPPRACTICE_LIST_H

#include <iostream>

template <class T>
class List {
private:
    int size_;
    T *ptr;

public:
    List(int s = 0);//s表示数组元素个数
    List(List & another);
    ~List();
    void push_bask(const T &v);

    //赋值号的作用是使"="左边对象里存放的数组，大小和内容都和右边的对象一样
    List & operator=(const List & another);

    int length(){ return size_; }
    //用以支持根据下标访问数组元素，如a[i] = 4;和n = a[i]这样的语句
    T & operator[](int i){
        return ptr[i];
    }
};

template<class T>
List<T>::List(int s):size_(s){
    if(s <= 0)
        ptr = nullptr;
    else
        ptr = new T[s];
}

template<class T>
List<T>::List(List & another){
    if(!another.ptr){
        ptr = nullptr;
        size_ = 0;
        return;
    }

    ptr = new T[another.size_];
    memcpy(ptr, another.ptr, sizeof(T) * another.size_);
    size_ = another.size_;
}

template <class T>
List<T>::~List(){
    if(ptr)
        delete []ptr;
}

template <class T>
List<T>& List<T>::operator=(const List<T> &another) {
    if(this == &another)
        return * this;
    if(another.ptr == nullptr){
        if (ptr)
            delete []ptr;
        ptr = nullptr;
        size_ = 0;
        return * this;
    }

    if(size_ < another.size_){
        if (ptr)
            delete []ptr;
        ptr = new T[another.size_];
    }

    memcpy(ptr, another.ptr, sizeof(T) * another.size_);
    size_ = another.size_;
    return * this;
}

template <class T>
void List<T>::push_bask(const T &v) {
    if (ptr)
    {
        T *temp_ptr = new T[size_ + 1];
        memcpy(temp_ptr, ptr, sizeof(T) * size_);
        delete []ptr;
        ptr = temp_ptr;
    }else{
        ptr = new T[1];
    }

    ptr[size_++] = v;
}

#endif //CPPRACTICE_LIST_H
