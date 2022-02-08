//
// Created by DEV on 2021/11/1.
//

#ifndef CPPRACTICE_PLAYER_H
#define CPPRACTICE_PLAYER_H

#include <string>
using namespace std;

class Player{
public:
    Player(string name, int score){
        this->m_score_ = score;
        this->m_name_ = name;
    };

    string m_name_;
    int m_score_;
};

class PlayerComparison{
public:
    //添加const后缀，避免警告：the specified comparator type does not provide a viable const call operator
    bool operator()(const Player& p1, const Player& p2) const{
        //降序
        return p1.m_score_ > p2.m_score_;
    }
};

#endif //CPPRACTICE_PLAYER_H
