//
// Created by DEV on 2021/12/27.
//

#ifndef CPPRACTICE_PERSON_H
#define CPPRACTICE_PERSON_H

#include <utility>

#include "string"

using namespace std;

class Person{
public:
    int age_;
    string name_;

    Person(int age, string name) :
            age_(age),
            name_(std::move(name)){

    }

    void ShowInfo(){
        std::cout << "age = " << age_ << ", name = " << name_ << std::endl;
    }
};

#endif //CPPRACTICE_PERSON_H
