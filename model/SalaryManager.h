//
// Created by DEV on 2022/1/29.
//

#ifndef CPPRACTICE_SALARYMANAGER_H
#define CPPRACTICE_SALARYMANAGER_H

#include "person.h"

class SalaryManager{
    Person *employers_;
    int max_;
    int count_;

public:
    explicit SalaryManager(int max = 0){
        max_ = max;
        count_ = 0;
        employers_ = new Person[max];
    }

    double &operator[](const string& name){
        Person *new_p;
        for(new_p = employers_; new_p < employers_ + count_; new_p++){
            if (new_p->name_ == name){
                return new_p->salary_;
            }
        }

        new_p = employers_ + count_++;
        new_p->name_ = name;
        new_p->salary_ = 0;
        return new_p->salary_;
    }

    ~SalaryManager(){
        if (employers_){
            delete []employers_;//‼️删除数组
            employers_ = nullptr;
        }
    }

    void display(){
        for (int i = 0; i < count_; ++i) {
            cout << employers_[i].name_ << "'s salary = $" << employers_[i].salary_ << endl;
        }
    }
};

#endif //CPPRACTICE_SALARYMANAGER_H
