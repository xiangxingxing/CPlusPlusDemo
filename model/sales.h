//
// Created by DEV on 2022/2/8.
//

#ifndef CPPRACTICE_SALES_H
#define CPPRACTICE_SALES_H

#include <iostream>
#include <string>

using namespace std;

class Sales{
private:
    char name_[10]{};
    char id_[18]{};
    int age_;

public:
    Sales(const char *name, const char *id, int age);

    friend Sales &operator<<(ostream &os, Sales &s);
    friend Sales &operator>>(istream &is, Sales &s);
};

Sales::Sales(const char *name, const char *id, int age) {
    strcpy(name_, name);
    strcpy(id_, id);
    age_ = age;
}

Sales& operator<<(ostream &os, Sales &s){
    os<<s.name_<<"\t";
    os<<s.id_<<"\t";
    os<<s.age_<<endl;
    return s;
}

Sales& operator>>(istream &is, Sales &s){
    cout<<"输入雇员的姓名、身份证号、年龄"<<endl;
    is>>s.name_>>s.id_>>s.age_;
    return s;
}

#endif //CPPRACTICE_SALES_H
