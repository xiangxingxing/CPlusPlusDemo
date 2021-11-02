//
// Created by DEV on 2021/11/2.
//

#ifndef UNTITLED_LRU_CACHE_H
#define UNTITLED_LRU_CACHE_H

#include "map"
using namespace std;

struct DLinkedNode {
    int key_, value_;
    DLinkedNode* next_;
    DLinkedNode(): key_(0), value_(0), next_(nullptr) {}
    DLinkedNode(int key, int value): key_(key), value_(value), next_(nullptr) {}
};

class LRUCache {
private:
    int size_;
    int capacity_;
    map<int, DLinkedNode*> key_to_pre_;
    DLinkedNode* dummy_;
    DLinkedNode* tail_;

public:
    LRUCache(int capacity);

    int get(int key);

    void put(int key, int value) ;

    void MoveToTail(int key);
};

#endif //UNTITLED_LRU_CACHE_H