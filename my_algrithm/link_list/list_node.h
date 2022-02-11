//
// Created by DEV on 2022/2/10.
//

#ifndef CPPRACTICE_LISTNODE_H
#define CPPRACTICE_LISTNODE_H

class ListNode{
public:
    int val;
    ListNode * next;
    ListNode(int val) {
        this->val = val;
        this->next = nullptr;
    }
};

#endif //CPPRACTICE_LISTNODE_H
