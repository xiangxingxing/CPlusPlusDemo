//
// Created by xiangxx on 2024/10/15.
//

#ifndef CPPRACTICE_2024_MS_REVIEW_H
#define CPPRACTICE_2024_MS_REVIEW_H

#include <vector>
#include <string>

using namespace std;

struct TreeNode{
	int val;
	TreeNode* left;
	TreeNode* right;
	explicit TreeNode(int x):val(x), left(nullptr),right(nullptr){}
	TreeNode():val(0), left(nullptr),right(nullptr){}
};

struct ListNode{
	int val;
	ListNode* next;
	explicit ListNode(int x): val(x), next(nullptr){}
	ListNode(): val(0), next(nullptr){}
	ListNode(int x, ListNode* next_): val(0), next(next_){}
};

class MS{
public:
	//1. Two Sum
	vector<int> twoSum(vector<int>& nums, int target);

	//region BinarySearch

	int binarySearch(vector<int>& nums, int target);

	int findMin(vector<int>& nums);

	int binarySearchRotated(vector<int>& nums, int target);

	//endregion

	//region Tree

	//98. Validate Binary Search Tree
	bool isValidBST(TreeNode* root);
	bool isValidBST(TreeNode* root, long low, long high);

	//99. Recover Binary Search Tree
	TreeNode* prev = nullptr;
	TreeNode* first = nullptr;
	TreeNode* second = nullptr;
	void recoverTree(TreeNode* root);
	void inorder(TreeNode* node);

	//145. Binary Tree Postorder Traversal
	vector<int> postorderTraversal(TreeNode* root);

	//235. Lowest Common Ancestor of a Binary Search Tree
	TreeNode* lowestCommonAncestorBST(TreeNode* root, TreeNode* p, TreeNode* q);

	//236. Lowest Common Ancestor of a Binary Tree
	TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q);


	//450. Delete Node in a BST
	TreeNode* deleteNode(TreeNode* root, int key);
	TreeNode* getMin(TreeNode* root);

	//endregion

	//region LinkedList

	//2. Add Two Numbers
	ListNode* addTwoNumbers(ListNode* l1, ListNode* l2);

	//19. Remove Nth Node From End of List
	ListNode* removeNthFromEnd(ListNode* head, int n);

	//21. Merge Two Sorted Lists
	ListNode* mergeTwoLists(ListNode* list1, ListNode* list2);

	//23. Merge k Sorted Lists
	ListNode* mergeKLists(vector<ListNode*>& lists);
	ListNode* mergeKListsUseMinHeap(vector<ListNode*>& lists);

	//24. Swap Nodes in Pairs
	ListNode* swapPairs(ListNode* head);

	//25. Reverse Nodes in k-Group
	ListNode* reverseKGroup(ListNode* head, int k);
	ListNode* reverseKGroupHelper(ListNode* head);

	//61. Rotate List
	ListNode* rotateRight(ListNode* head, int k);
	ListNode* rotateRightHelper(ListNode* head, int k);

	//189. Rotate Array
	void rotate(vector<int>& nums, int k);

	//82. Remove Duplicates from Sorted List II
	ListNode* deleteDuplicates2(ListNode* head);

	//83. Remove Duplicates from Sorted List
	ListNode* deleteDuplicates(ListNode* head);

	//141. Linked List Cycle
	bool hasCycle(ListNode *head);

	//142. Linked List Cycle II
	ListNode *detectCycle(ListNode *head);

	//148. Sort List
	ListNode* sortList(ListNode* head);
	ListNode* mergeTwoSorted(ListNode* l1, ListNode* l2);

	//206. Reverse Linked List
	ListNode* reverseList(ListNode* head);


	//endregion

	//region Dynamic Programming 动态规划
	/**
	 * 动态规划
	 *      1.确定状态【最后一步 --> 子问题】
	 *      2.转移方程
	 *      3.初始条件、边界
	 *      4.计算顺序
	 *
	 * 特点：
	 *      1.计数
	 *      2.求最大/最小值
	 *      3.求存在性
	 * */

	//5.Longest Palindromic Substring
	string longestPalindrome(const std::string &s);
	int findLongestPalindrome(const std::string &s, int i, int j);

	//22. Generate Parentheses
	vector<string> generateParenthesis(int n);
	void back_tracking(int open, int close, string& current, vector<string>& result);

	//55. JumpGame
	bool canJump(vector<int>& nums);

	//42. Trapping Rain Water
	int trap(vector<int>& height);

	//45. JumpGame II
	int jump(vector<int>& nums);

	//53. Maximum Subarray
	int maxSubArray(vector<int>& nums);

	//62. Unique Paths 【坐标型coordinator type】
	int uniquePaths(int m, int n);

	//63. Unique Paths II
	int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid);

	//64. Minimum Path Sum
	int minPathSum(vector<vector<int>>& grid);

	//115. Distinct Subsequences
	int numDistinct(string s, string t);

	//198.House Robber
	int rob(vector<int>& nums);

	//213. House Robber II
	int robCircle(vector<int>& nums);

	//377. Combination Sum IV
	int combinationSum4(vector<int>& nums, int target);

	/*
	 * k数之和
     * 输入：24，找出四个数字之和 = 24    输出：47种方案
	 * */
	int kSum(vector<int>& nums, int k, int target);

	//endregion

	//region Sort

	void quick_sort(vector<int>& array);
	void quick_sort(vector<int>& array, int start, int end);
	int internal_quick_sort(vector<int>& array, int start, int end);

	void insert_sort(vector<int>& array);

	void merge_sort(vector<int>& array);
	void merge_sort(vector<int>& array, int start, int end, vector<int>& temp);
	void internal_merge_sort(vector<int>& array, int start, int mid, int end, vector<int>& temp);

	void heap_sort(vector<int>& array);
	void heap_adjust(vector<int>& array, int index, int length);
	void heap_adjust_non_recursive(vector<int>& array, int index, int length);

	//347. Top K Frequent Elements
	vector<int> topKFrequent(vector<int>& nums, int k);

	//endregion
};

#endif //CPPRACTICE_2024_MS_REVIEW_H
