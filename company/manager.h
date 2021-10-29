//
// Created by DEV on 2021/10/29.
//

#ifndef UNTITLED_MANAGER_H
#define UNTITLED_MANAGER_H

#include "worker.h"

class Manager : public Worker{
public:
    Manager(int id, string name, int dept_id);

    virtual void ShowInfo();
    virtual string GetDeptName();
};

#endif //UNTITLED_MANAGER_H
