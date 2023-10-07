//
// Created by xiangxx on 2023/8/17.
//

#ifndef CPPRACTICE_LINTCODEMANAGER_H
#define CPPRACTICE_LINTCODEMANAGER_H

#include <stack>
using namespace std;

class LintCodeManager{
public:
	int binarySearch(vector<int> &nums, int target);

	//LintCode15.全排列
	vector<vector<int>> permute(vector<int> &nums);
	void helper(vector<int>& nums, vector<bool>& visited, vector<int>& temp, vector<vector<int>>& result);

	/*
	 * LintCode464、mergeSort
	 * */
	void mergeSort(vector<int> &a);
	void mergeSort(vector<int> &a, int start, int end, vector<int>& temp);
	void internalMerge(vector<int> &a, int start, int end, vector<int>& temp);

	/*
	 * quickSort
	 * */
	void quickSort(vector<int>& a);
	void quickSort(vector<int>& a, int start, int end);
	int internalQuickSort(vector<int>& a, int start, int end);

	/*
	 * heapSort
	 * */
	void heapSort(vector<int>& a);
	void heapAdjust(vector<int>& nums, int index, int len);

	/*
	 * LintCode92 · 背包问题
	 * */
	int backPack(int m, vector<int> &a);

	/*
	 * LintCode125 · 背包问题（二）
	 * */
	int backPackII(int m, vector<int> &a, vector<int> &v);

	/*
	 * LintCode5 · 第k大元素
	 * */
	int kthLargestElement(int k, vector<int> &nums);
	int kthHelper(vector<int> &nums, int start, int end, int targetIndex);

	/*
	 * 90.k数和2
	 * */
	vector<vector<int>> kSumII(vector<int> &a, int k, int target);
	void kSumIIHelper(vector<int>& a, int offset, int remainCount, int remainTarget, vector<int>& temp, vector<vector<int>>& result);
};

//LintCode40 · 用栈实现队列
class MyQueue {

private:
	std::stack<int>* s1;
	std::stack<int>* s2;

public:
	MyQueue() {
		// do initialization if necessary
		s1 = new std::stack<int>();
		s2 = new std::stack<int>();
	}

	/*
	 * @param element: An integer
	 * @return: nothing
	 */
	void push(int element) {
		// write your code here
		s1->push(element);
	}

	/*
	 * @return: An integer
	 */
	int pop() {
		// write your code here
		top();
		int res = s2->top();
		s2->pop();
		return res;
	}

	/*
	 * @return: An integer
	 */
	int top() {
		// write your code here
		if(s2->empty()){
			while(!s1->empty()){
				s2->push(s1->top());
				s1->pop();
			}
		}

		return s2->top();
	}
};

class MinStack {

	std::stack<int> sk;
	std::stack<int> minSk;
public:
	MinStack() {
		// do intialization if necessary
	}

	/*
	 * @param number: An integer
	 * @return: nothing
	 */
	void push(int number) {
		// write your code here
		sk.push(number);

		if(minSk.empty()){
			minSk.push(number);
		}
		else{
			minSk.push(minSk.top() < number ? minSk.top() : number);
		}
	}

	/*
	 * @return: An integer
	 */
	int pop() {
		// write your code here
		if(!sk.empty() && !minSk.empty()){
			int res = sk.top();
			sk.pop();
			minSk.pop();
			return res;
		}

		throw;
	}

	/*
	 * @return: An integer
	 */
	int min() {
		// write your code here
		if(minSk.empty()){
			throw;
		}

		return minSk.top();
	}
};

#endif //CPPRACTICE_LINTCODEMANAGER_H
