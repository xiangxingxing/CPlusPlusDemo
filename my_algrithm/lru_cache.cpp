//
// Created by DEV on 2021/11/2.
//

#include "lru_cache.h"
#include <map>
using namespace std;

/*
 * 适用场景建议
	单链表（带 keyToPrev）适用场景：

	内存敏感的场景：如果 LRU 缓存的容量较大，单链表的空间开销较低。
	简单实现的需求：实现方式相对简单，使用哈希表和单链表即可完成。
	双链表适用场景：

	高性能需求：双链表更加灵活，维护链表结构更简洁直接。
	内存不是瓶颈：如果 LRU 缓存容量不大，双链表额外的指针开销可以忽略
 * */

LRUCache::LRUCache(int capacity) {
    capacity_ = capacity;
    size_ = 0;
    dummy_ = new DLinkedNode();
    tail_ = dummy_;
}

int LRUCache::get(int key) {
    if (key_to_pre_.find(key) == key_to_pre_.end()){
        return -1;
    }

    MoveToTail(key);
    return tail_->value_;
}

void LRUCache::put(int key, int value) {
    if (key_to_pre_.find(key) != key_to_pre_.end()){
        auto pre_node = key_to_pre_[key];
        pre_node->next_->value_ = value;
        MoveToTail(key);
        return;
    }

    if (size_ < capacity_){
        DLinkedNode* new_node = new DLinkedNode(key, value);
//        new_node->next_ = nullptr;
        tail_->next_ = new_node;
        key_to_pre_[key] = tail_;
        tail_ = new_node;
        size_++;
        return;
    }

    auto first = dummy_->next_;
    key_to_pre_.erase(first->key_);
    first->key_ = key;
    first->value_ = value;
    key_to_pre_[key] = dummy_;
    MoveToTail(key);
}

void LRUCache::MoveToTail(int key) {
    map<int, DLinkedNode*>::iterator iterator = key_to_pre_.find(key);
    auto pre_node = iterator->second;
    DLinkedNode* cur_node = pre_node->next_;
    if (cur_node == tail_){
        return;
    }
    pre_node->next_ = cur_node->next_;
	cur_node->next_ = nullptr;
    tail_->next_ = cur_node;
    key_to_pre_[key] = tail_;
    key_to_pre_[pre_node->next_->key_] = pre_node;
    tail_ = cur_node;
}