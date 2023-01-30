//
// Created by DEV on 2022/2/10.
//

#include "link_list_manager.h"

/*
 * https://www.lintcode.com/problem/35/
 * 翻转链表
 * */
ListNode * LinkListManager::reverse(ListNode * head){
    ListNode * cur = head;
    ListNode * pre = nullptr;
    while(cur != nullptr){
        ListNode * next = cur->next;
        cur->next = pre;
        pre = cur;
        cur = next;
    }

    return pre;
}

/*
 * https://www.lintcode.com/problem/452/
 * 删除链表中等于给定值 val 的所有节点
 * */
ListNode * LinkListManager::removeElements(ListNode * head, int val){
    ListNode * dummy = new ListNode(0);
    dummy->next = head;
    ListNode * cur = dummy;
    while(cur != nullptr && cur->next != nullptr){
        if(cur->next->val == val){
            cur->next = cur->next->next;
        }else{
            cur = cur->next;
        }
    }

    return dummy->next;
}

/*
 * https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/
 * 19. 删除链表的倒数第 N 个结点
 * */
ListNode* LinkListManager::removeNthFromEnd(ListNode* head, int n){
    ListNode * dummy = new ListNode(0);
    dummy->next = head;
    ListNode * pre = dummy;
    ListNode * post = dummy;
    while (n > 0){
        post = post->next;
        n--;
    }
    while (post != nullptr && post->next != nullptr){
        pre = pre->next;
        post = post->next;
    }

    pre->next = pre->next->next;
    return dummy->next;
}

ListNode* LinkListManager::deleteDuplicates(ListNode* head){
    if(head == nullptr || head->next == nullptr){
        return head;
    }
    ListNode* p = head;
//    while (p != nullptr){
//        ListNode * q = p->next;
//        while (q != nullptr && q->val == p->val){
//            q = q->next;
//        }
//        p->next = q;
//        p = p->next;
//    }
    while (p != nullptr && p->next != nullptr){
        if(p->val == p->next->val){
            p->next = p->next->next;
        }else{
            p = p->next;
        }
    }

    return head;
}

/*
 * https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/
 * 删除排序链表中的重复元素 II 只留下不同的数字
 * */
ListNode* LinkListManager::deleteDuplicates2(ListNode* head){
    if(head == nullptr || head->next == nullptr){
        return head;
    }

    auto* dummy = new ListNode(0);
    dummy->next = head;
    ListNode * cur = dummy;
    while (cur->next != nullptr && cur->next->next != nullptr){
        if(cur->next->val == cur->next->next->val){
            auto val = cur->next->val;
            while (cur->next != nullptr && cur->next->val == val){
                cur->next = cur->next->next;
            }
        }else{
            cur = cur->next;
        }
    }

    return dummy->next;
}

bool LinkListManager::hasCycle(ListNode * head) {
	// write your code here
	if(head == nullptr || head->next == nullptr) return false;
	ListNode* slow = head;
	ListNode* fast = head;
	while(fast != nullptr && fast->next != nullptr){
		fast = fast->next->next;
		slow = slow->next;
		if(fast == slow){
			return true;
		}
	}
	return false;
}

ListNode* LinkListManager::detectCycle(ListNode *head) {
	if(head == nullptr || head->next == nullptr) return nullptr;
	ListNode* p = nodeOnCycle(head);
	ListNode* q = head;
	while(p != nullptr && q != nullptr){
		if(p == q){
			break;
		}
		p = p->next;
		q = q->next;
	}

	return p;
}

ListNode* LinkListManager::nodeOnCycle(ListNode * head) {
	// write your code here
	if(head == nullptr || head->next == nullptr) return nullptr;
	ListNode* slow = head;
	ListNode* fast = head;
	while(fast != nullptr && fast->next != nullptr){
		fast = fast->next->next;
		slow = slow->next;
		if(fast == slow){
			return fast;
		}
	}
	return nullptr;
}

ListNode* LinkListManager::mergeKLists(vector<ListNode*> &lists){
	if(lists.empty()){
		return nullptr;
	}

	if(lists.size() == 1){
		return lists.at(0);
	}

	for (size_t i = lists.size() - 1; i > 0; i--)
	{
		lists[0] = mergeTwo(lists.at(i), lists.at(0));
	}

	return lists[0];
}

ListNode* LinkListManager::mergeTwo(ListNode* l1, ListNode* l2){
	if(l1 == nullptr && l2 == nullptr) return nullptr;
	if(l1 == nullptr) return l2;
	if(l2 == nullptr) return l1;

	auto* dummy = new ListNode(0);
	ListNode* p = dummy;
	while (l1 != nullptr && l2 != nullptr){
		if(l1->val < l2->val){
			p->next = l1;
			l1 = l1->next;
		}
		else{
			p->next = l2;
			l2 = l2->next;
		}
		p = p->next;
	}
	p->next = l1 != nullptr ? l1 : l2;

	return dummy->next;
}