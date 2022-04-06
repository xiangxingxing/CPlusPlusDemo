//
// Created by DEV on 2021/11/1.
//
#include "std_manager.h"
#include "example/player.h"

#include <iostream>
#include "vector"
#include <string>
#include <deque>
#include <algorithm>
#include <ctime>
#include "set"

using namespace std;

void StdManager::VectorTraversal() {
    vector<int> v;
    v.push_back(10);
    v.push_back(20);
    v.push_back(30);
    v.push_back(40);

    vector<int>::iterator begin = v.begin();
    vector<int>::iterator end = v.end();
    //第一种遍历
//    while (begin != end){
//        cout << *begin << endl;
//        begin++;
//    }

    //第二种遍历
//    for (vector<int>::iterator it = integers_.begin(); it != integers_.end() ; it++) {
//        cout << *it << endl;
//    }

    //第三种遍历
    for_each(v.begin(), v.end(), my_print);
}

void StdManager::my_print(int val) {
    cout << val << endl;
}

void StdManager::StringConstructDemo() {
    string s1;//默认构造

    const char * str = "hello world";//c语言风格的字符串
    string s2(str);
    cout << "s2 = " << s2 << endl;

    string s3(s2);
    cout << "s3 = " << s3 << endl;

    string s4(10, 'a');
    cout << "s4 = " << s4 << endl;

    string s5;
    s5.assign("hello levi", 5);
    cout << "s5 = " << s5 << endl;

    string s6;
    s6.assign(6, 'q');
    cout << "s6 = " << s6 << endl;

}

void StdManager::VectorFuncDemo() {
    vector<int> v;
    v.reserve(100000);
    int num = 0;//统计开辟次数
    int * p = nullptr;
    for (int i = 0; i < 100000; ++i) {
        v.push_back(i);
        if(p != &v[0]){
            p = &v[0];
            num++;
        }
    }
    cout << "reserve预留空间测试：vector内存重新分配了" << num << "次" << endl;
}


void StdManager::PlayerDemo() {
    //随机数种子
    srand((unsigned int) time(nullptr));

    vector<Player> v;
    CreatePlayer(v);
    SetScore(v);
    ShowScore(v);
}

void StdManager::CreatePlayer(vector<Player> &v) {
    string name_seed = "ABCDE";
    for (int i = 0; i < name_seed.length(); ++i) {
        string name = "选手";
        name += name_seed[i];
        int score = 0;
        Player p(name, score);
        v.push_back(p);
    }
}

void StdManager::SetScore(vector<Player> &v) {
    for (vector<Player>::iterator it = v.begin(); it != v.end() ; it++) {
        deque<int> d;
        for (int i = 0; i < 10; ++i) {
            int score = rand() %41 + 60;
            d.push_back(score);
        }

        sort(d.begin(), d.end());
        d.pop_front();
        d.pop_back();

        int sum = 0;
        for(auto & elem : d){
            sum += elem;
        }

        int avg = sum / d.size();

        it->m_score_ = avg;
    }
}

void StdManager::ShowScore(vector<Player> &v) {
    for(auto & elem : v){
        cout << "姓名: " << elem.m_name_ << "  平均分为： " << elem.m_score_ <<endl;
    }
}

void StdManager::SetSortDemo() {
    Player p1("levi", 18);
    Player p2("aqua", 29);
    Player p3("blom", 30);
    Player p4("dog", 22);

    set<Player, PlayerComparison>s;
    s.insert(p1);
    s.insert(p2);
    s.insert(p3);
    s.insert(p4);

    for(set<Player, PlayerComparison>::iterator it = s.begin(); it != s.end(); it++){
        cout << it->m_name_ << "的分数为 " << it->m_score_ << endl;
    }
}
