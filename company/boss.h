//
// Created by DEV on 2021/10/29.
//

#ifndef UNTITLED_BOSS_H
#define UNTITLED_BOSS_H

#include "worker.h"

class Boss : public Worker{
public:
    Boss(int id, string name, int dept_id);

    virtual void ShowInfo();
    virtual string GetDeptName();
};


#endif //UNTITLED_BOSS_H
