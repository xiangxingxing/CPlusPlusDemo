//
// Created by DEV on 2021/10/29.
//

#ifndef CPPRACTICE_MANAGER_H
#define CPPRACTICE_MANAGER_H

#include "worker.h"

class Manager : public Worker{
public:
    Manager(int id, string name, int dept_id);

    virtual void ShowInfo();
    virtual string GetDeptName();
};

#endif //CPPRACTICE_MANAGER_H
