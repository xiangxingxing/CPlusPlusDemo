//
// Created by DEV on 2021/10/27.
//

#ifndef CPPRACTICE_STUDENT_H
#define CPPRACTICE_STUDENT_H

#include "iostream"
using namespace std;

class Student {
public:
    Student(string name, int age, float score);
    //拷贝构造函数:只能有一个本类型的常引用参数
    Student(const Student &stu);
    void show();
    string GetName() const;
    int GetAge() const;
    void SetAge(int age);
    float GetScore() const;
    friend void DisPlay(Student &stu);

public:
    static int GetTotal();
    static float GetPoints();

private:
    static int k_total_;
    static float k_points_;

private:
    string m_name_;
    int m_age_;
    float m_score_;
};

struct CompareAge{
	bool operator()(const Student& s1, const Student& s2){
		return s1.GetAge() > s2.GetAge();//最小堆
	}
};

#endif //CPPRACTICE_STUDENT_H
