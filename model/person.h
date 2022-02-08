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

    double salary_;

    Person(){
        age_ = 0;
        name_ = "";
        salary_ = 0;
    }

    Person(int age, string name, double salary = 0.0) :
            age_(age),
            name_(std::move(name)),
            salary_(salary){

    }

    ~Person()= default;

    void ShowInfo() const{
        std::cout << "age = " << age_ << ", name = " << name_ << std::endl;
    }
};

#endif //CPPRACTICE_PERSON_H
