//
// Created by DEV on 2022/2/10.
//

#ifndef CPPRACTICE_LINK_LIST_MANAGER_H
#define CPPRACTICE_LINK_LIST_MANAGER_H

#include "list_node.h"
#include "vector"

using namespace std;

class LinkListManager{
public:
    ListNode * reverse(ListNode * head);
    ListNode * removeElements(ListNode * head, int val);
    ListNode* removeNthFromEnd(ListNode* head, int n);
    ListNode* deleteDuplicates(ListNode* head);
    ListNode* deleteDuplicates2(ListNode* head);

	bool hasCycle(ListNode * head);
	ListNode *detectCycle(ListNode *head);
	ListNode* nodeOnCycle(ListNode * head);

	ListNode *mergeKLists(vector<ListNode*> &lists);
	ListNode* mergeTwo(ListNode* l1, ListNode* l2);
};

#endif //CPPRACTICE_LINK_LIST_MANAGER_H
