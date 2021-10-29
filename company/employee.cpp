//
// Created by DEV on 2021/10/29.
//

#include "employee.h"

Employee::Employee(int id, string name, int dept_id) {
    this->m_id_ = id;
    this->m_name_ = name;
    this->m_dept_id_ = dept_id;
}

void Employee::ShowInfo(){
    cout << "职工编号： " << this->m_id_
        << " \t职工姓名：  " << this->m_name_
        << " \t岗位：  " << this->GetDeptName()
        << " \t 岗位职责：完成经理的任务" << endl;
}

string Employee::GetDeptName(){
    return string("普通员工");
}