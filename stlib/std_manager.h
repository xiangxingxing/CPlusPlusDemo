//
// Created by DEV on 2021/11/1.
//

#ifndef CPPRACTICE_STD_MANAGER_H
#define CPPRACTICE_STD_MANAGER_H

#include "example/player.h"
#include "vector"
using namespace std;

class StdManager{
public:
    static void my_print(int val);
    static void VectorDemo();
    static void VectorFuncDemo();
    static void StringConstructDemo();
    static void SetSortDemo();

    static void PlayerDemo();
    static void CreatePlayer(vector<Player> &v);
    static void SetScore(vector<Player> &v);
    static void ShowScore(vector<Player> &v);
};

#endif //CPPRACTICE_STD_MANAGER_H
