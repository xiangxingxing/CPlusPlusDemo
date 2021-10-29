//
// Created by DEV on 2021/10/29.
//

#include "boss.h"

Boss::Boss(int id, string name, int dept_id) {
    this->m_id_ = id;
    this->m_name_ = name;
    this->m_dept_id_ = dept_id;
}

void Boss::ShowInfo(){
    cout << "职工编号： " << this->m_id_
         << " \t职工姓名：  " << this->m_name_
         << " \t岗位：  " << this->GetDeptName()
         << " \t 岗位职责：管理公司" << endl;
}

string Boss::GetDeptName(){
    return string("总裁");
}

