//
// Created by DEV on 2021/10/29.
//

#ifndef UNTITLED_EMPLOYEE_H
#define UNTITLED_EMPLOYEE_H

#include "worker.h"

class Employee : public Worker{
public:
    Employee(int id, string name, int dept_id);

    virtual void ShowInfo();
    virtual string GetDeptName();
};


#endif //UNTITLED_EMPLOYEE_H
