//
// Created by DEV on 2021/11/1.
//

#ifndef UNTITLED_TREE_NODE_H
#define UNTITLED_TREE_NODE_H

#include "vector"
using namespace std;

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

#endif //UNTITLED_TREE_NODE_H