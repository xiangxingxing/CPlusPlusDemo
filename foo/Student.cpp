//
// Created by DEV on 2021/10/27.
//

#include "Student.h"
#include "iostream"
using namespace std;

int Student::k_total_ = 0;//初始化静态变量
float Student::k_points_ = 0.0;//初始化静态变量

//构造函数初始化列表
Student::Student(string name, int age, float score) :
m_name_(name), m_age_(age), m_score_(score){
    k_total_++;
    k_points_ += score;
}

void Student::show(){
    cout<<m_name_<<"的年龄是"<<m_age_<<",成绩是"<<m_score_<<"有"<<k_total_<<"名学生"<<endl;
}

float Student::GetPoints() {
    return k_points_;
}

int Student::GetTotal() {
    return k_total_;
}

int Student::GetAge() const{
    return m_age_;
}

string Student::GetName() const{
    return m_name_;
}

float Student::GetScore() const{
    return m_score_;
}

//全局范围内的非成员函数，不属于任何类
void DisPlay(Student *pstu) {
    cout<<pstu->m_name_<<" 的年龄是 "<<pstu->m_age_<<",成绩是 "<<pstu->m_score_<<endl;
}

Student::Student(const Student &stu) {
    m_name_ = stu.m_name_;
    m_score_ = stu.m_score_;
    m_age_ = stu.m_age_;
}
