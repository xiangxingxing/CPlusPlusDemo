//
// Created by DEV on 2021/10/29.
//

#ifndef CPPRACTICE_WORKER_H
#define CPPRACTICE_WORKER_H

#include "iostream"
#include <string>

using namespace std;

class Worker{
public:
    virtual void ShowInfo() = 0;
    virtual string GetDeptName() = 0;

    int m_id_;
    string m_name_;
    int m_dept_id_;//部门名称编号
};

#endif //CPPRACTICE_WORKER_H
