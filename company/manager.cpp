//
// Created by DEV on 2021/10/29.
//

#include "manager.h"

Manager::Manager(int id, string name, int dept_id) {
    this->m_id_ = id;
    this->m_name_ = name;
    this->m_dept_id_ = dept_id;
}

void Manager::ShowInfo(){
    cout << "职工编号： " << this->m_id_
         << " \t职工姓名：  " << this->m_name_
         << " \t岗位：  " << this->GetDeptName()
         << " \t 岗位职责：完成老板交给的任务，并下发任务给员工" << endl;
}

string Manager::GetDeptName(){
    return string("经理");
}

