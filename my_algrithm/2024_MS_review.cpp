//
// Created by xiangxx on 2024/10/15.
//

#include "2024_MS_review.h"

#include <vector>
#include <stack>
#include <queue>
#include <unordered_map>
#include <string>

using namespace std;

//1. Two Sum
vector<int> MS::twoSum(vector<int>& nums, int target){
	vector<int> result(2, -1);
	unordered_map<int, int> map;
	for (int i = 0; i < nums.size(); ++i)
	{
		int complement = target - nums[i];
		if(map.find(complement) != map.end()){
			result[0] = i;
			result[1] = map[complement];
			break;
		}

		map[nums[i]] = i;
	}

	return result;
}

//region Binary Search
/*
 * LeetCode 704: Binary Search
 * left + 1 < right：这种条件的好处是可以避免死循环，
 * 并且当搜索空间收缩到只剩下两个元素时，退出循环并直接对 left 和 right 进行判断。
 * 这种方法尤其适用于某些需要精确控制边界情况的题目。
 *
 * Time Complexity = O(log n)
 * Space Complexity = O(1)
 * */
int MS::binarySearch(vector<int>& nums, int target)
{
	int low = 0;
	int high = (int)nums.size() - 1;
	while(low + 1 < high){
		int mid = low + (high - low) / 2;
		if(nums[mid] == target){
			return mid;
		}
		else if(nums[mid] > target){
			high = mid;
		}
		else{
			low = mid;
		}
	}

	if(nums[low] == target) return low;
	if(nums[high] == target) return high;
	return -1;
}

/*
 * LeetCode 153: Find Minimum in Rotated Sorted Array
 * */
int MS::findMin(vector<int>& nums){
	int low = 0;
	int high = (int)nums.size() - 1;
	while(low + 1 < high){
		int mid = low + (high - low) / 2;
		if(nums[mid] > nums[high]){
			low = mid;
		}
		else{
			high = mid;
		}

//		if(nums[mid] > nums[low]){
//			if(nums[mid] > nums[high]){
//				low = mid;
//			}
//			else{
//				high = mid;
//			}
//		}
//		else if(nums[mid] < nums[high]){
//			if(nums[mid] < nums[low]){
//				high = mid;
//			}
//			else{
//				low = mid;
//			}
//		}
	}

	return std::min(nums[low], nums[high]);
}

/*
 * LeetCode 33: Search in Rotated Sorted Array
 * */
int MS::binarySearchRotated(vector<int>& nums, int target)
{
	int low = 0;
	int high = (int)nums.size() - 1;
	while(low + 1 < high){
		int mid = low + (high - low) / 2;
		if(nums[mid] == target){
			return mid;
		}
		if(nums[low] <= nums[mid]){
			if(nums[low] <= target && target < nums[mid]){
				high = mid;
			}
			else{
				low = mid;
			}
		}
		else{
			if(nums[mid] < target && target <= nums[high]){
				low = mid;
			}
			else{
				high = mid;
			}
		}
	}

	if(nums[low] == target) return low;
	if(nums[high] == target) return high;

	return -1;
}

//endregion

//region Tree

//98. Validate Binary Search Tree
// 时间复杂度 O(n),n为树中的节点数量
// 空间复杂度 O(h),h为树的高度：平衡的二叉树，高度为O(log n)，非平衡二叉树，最坏下为链状树，高度为O(n)
bool MS::isValidBST(TreeNode* root) {
	return isValidBST(root, LONG_MIN, LONG_MAX);
}

bool MS::isValidBST(TreeNode* root, long low, long high){
	if(root == nullptr) return true;
	if(root->val <= low || root->val >= high) return false;
	return isValidBST(root->left, low, root->val) && isValidBST(root->right, root->val, high);
}

//99. Recover Binary Search Tree
//时间复杂度：O(N)，其中 N 是树中的节点数量。
//空间复杂度：O(H)，其中 H 是树的高度。
// 		空间复杂度主要取决于递归调用栈的深度。
//		在最坏情况下（链状树），空间复杂度为 O(N)；而对于平衡树，空间复杂度为 O(log N)。
void MS::recoverTree(TreeNode* root){
	if(!root) return;
	inorder(root);
	if(first && second){
		swap(first->val, second->val);
	}
}

void MS::inorder(TreeNode* node){
	if(!node) return;
	inorder(node->left);
	if(prev && node->val < prev->val){
		if(!first){
			first = prev;
		}

		second = node;
	}
	prev = node;
	inorder(node->right);
}

//145. Binary Tree Postorder Traversal
vector<int> MS::postorderTraversal(TreeNode* root) {
	vector<int> result;
	if(!root) return result;
	stack<TreeNode*> sk;
	TreeNode* p = root;
	TreeNode* lastVisited = nullptr;
	while(p || !sk.empty()){
		if(p){
			sk.push(p);
			p = p->left;
		}
		else if(!sk.empty()){
			p = sk.top();
			if(!p->right || p->right == lastVisited){
				result.push_back(p->val);
				sk.pop();
				lastVisited = p;
				p = nullptr;
			}
			else{
				p = p->right;
			}
		}
	}

	return result;
}

//235. Lowest Common Ancestor of a Binary Search Tree
TreeNode* MS::lowestCommonAncestorBST(TreeNode* root, TreeNode* p, TreeNode* q){
	if(!root || root == p || root == q) return root;
	if(root->val > p->val && root->val > q->val){
		return lowestCommonAncestorBST(root->left, p, q);
	}
	if(root->val < p->val && root->val < q->val){
		return lowestCommonAncestorBST(root->right, p, q);
	}

	return root;
}

//236. Lowest Common Ancestor of a Binary Tree
TreeNode* MS::lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q){
	if(!root || root == p || root == q) return root;
	TreeNode* left = lowestCommonAncestor(root->left, p, q);
	TreeNode* right = lowestCommonAncestor(root->right, p, q);
	if(left && right){
		return root;
	}

	return left ? left : right;
}

//450. Delete Node in a BST
/*
 * 时间复杂度：O(H)，其中 H 是树的高度。
		在平衡的 BST 中，H = O(log N)，最坏情况下（链状树）为 O(N)。
   空间复杂度：O(H)，递归调用栈的深度。
 * */
TreeNode* MS::deleteNode(TreeNode* root, int key){
	if(!root) return root;
	if(root->val == key){
		// TreeNode* temp = root->right;
		// delete root;
		// return temp;
		if(!root->left) return root->right;
		if(!root->right) return root->left;
		TreeNode* minInRight = getMin(root->right);
		root->val = minInRight->val;
		root->right = deleteNode(root->right, minInRight->val);
	}
	else if (root->val < key){
		root->right = deleteNode(root->right, key);
	}
	else{
		root->left = deleteNode(root->left, key);
	}

	return root;
}

TreeNode* MS::getMin(TreeNode* root){
	TreeNode* p = root;
	while (p->left){
		p = p->left;
	}

	return p;
}

//endregion

//region Linked List

//2. Add Two Numbers
// Time/Space complexity: O(max(m, n))
ListNode* MS::addTwoNumbers(ListNode* l1, ListNode* l2){
	int pop = 0;
	ListNode* dummy = new ListNode();
	ListNode* cur = dummy;
	while(l1 || l2 || pop != 0){
		int add1 = l1 ? l1->val : 0;
		int add2 = l2 ? l2->val : 0;
		int sum = add1 + add2 + pop;
		int val = sum % 10;
		pop = sum / 10;
		cur->next = new ListNode(val);
		cur = cur->next;

		l1 = l1 ? l1->next : nullptr;
		l2 = l2 ? l2->next : nullptr;
	}

	return dummy->next;
}

//19. Remove Nth Node From End of List
ListNode* MS::removeNthFromEnd(ListNode* head, int n){
	ListNode* dummy = new ListNode();
	dummy->next = head;
	ListNode* firstNode = dummy;
	ListNode* secondNode = dummy;
	for(int i = 0; i <= n; i++){
		secondNode = secondNode->next;
	}
	while(secondNode){
		firstNode = firstNode->next;
		secondNode = secondNode->next;
	}

	ListNode* toDelete = firstNode->next;
	firstNode->next = firstNode->next->next;
	delete toDelete;

	return dummy->next;
}

//21. Merge Two Sorted Lists
// Time Complexity: O(m+n)
ListNode* MS::mergeTwoLists(ListNode* p, ListNode* q){
	ListNode* dummy = new ListNode();
	ListNode* cur = dummy;
	while(p && q){
		if(p->val <= q->val){
			cur->next = p;
			p = p->next;
		}
		else{
			cur->next = q;
			q = q->next;
		}

		cur = cur->next;
	}
	cur->next = p ? p : q;

	return dummy->next;
}

//23. Merge k Sorted Lists
ListNode* MS::mergeKLists(vector<ListNode*>& lists){
	if(lists.empty()) return nullptr;
	if(lists.size() == 1) return lists[0];
	for(int i = 1; i < lists.size(); i++){
		lists[0] = mergeTwoLists(lists[0], lists[i]);
	}

	return lists[0];
}

//最小堆：Time complexity: O(n* log k) ,k为链表数，n为节点数
//Space Complexity: O(k) 堆的大小最多为k
ListNode* MS::mergeKListsUseMinHeap(vector<ListNode*>& lists)
{
	auto compare = [](ListNode* n1, ListNode* n2){
		return n1->val > n2->val;//小根堆，值小的优先(因为默认大根堆，采用如int的默认 < 比较符)
	};
	priority_queue<ListNode*, vector<ListNode*>, decltype(compare)> minHeap(compare);
	for(auto& list : lists){
		if(list){
			minHeap.push(list);
		}
	}
	ListNode* dummy = new ListNode();
	ListNode* tail = dummy;
	while(!minHeap.empty()){
		ListNode* smallest = minHeap.top();
		minHeap.pop();
		tail->next = smallest;
		tail = tail->next;
		if(smallest->next){
			minHeap.push(smallest->next);
		}
	}

	ListNode* newHead = dummy->next;
	delete dummy;
	return newHead;
}

//24. Swap Nodes in Pairs
ListNode* MS::swapPairs(ListNode* head) {
	ListNode* dummy = new ListNode(0, head);
	ListNode* p = dummy;
	while(p->next && p->next->next){
		ListNode* first = p->next;
		ListNode* second = p->next->next;
		ListNode* next = second->next;
		second->next = first;
		first->next = next;
		p->next = second;
		p = first;
	}
	ListNode* newHead = dummy->next;
	delete dummy;
	return newHead;
}

//25. Reverse Nodes in k-Group
ListNode* MS::reverseKGroup(ListNode* head, int k){
	if(!head || !head->next || k == 1) return head;
	ListNode* dummy = new ListNode(0, head);
	//ListNode dummy(0, head);
	ListNode* tail = head;
	ListNode* cur = head;
	for (int i = 1; i < k && cur; ++i)
	{
		cur = cur->next;
	}
	if(!cur){
		return dummy->next;
	}
	ListNode* next = cur->next;
	cur->next = nullptr;
	dummy->next = reverseKGroupHelper(tail);
	tail->next = reverseKGroup(next, k);

	return dummy->next;
}

ListNode* MS::reverseKGroupHelper(ListNode* head){
	ListNode* pre = nullptr;
	ListNode* cur = head;
	while (cur){
		ListNode* next = cur->next;
		cur->next = pre;
		pre = cur;
		cur = next;
	}

	return pre;
}

//61. Rotate List
ListNode* MS::rotateRight(ListNode* head, int k){
	if(!head) return nullptr;
	ListNode* p = head;
	int length = 0;
	while(p){
		length++;
		p = p->next;
	}

	return rotateRightHelper(head, k % length);
}

ListNode* MS::rotateRightHelper(ListNode* head, int k){
	if(!head || k == 0 || !head->next) return head;
	ListNode* left = head;
	ListNode* right = head;
	for (int i = 0; i < k; ++i)
	{
		right = right->next;
	}

	while(right && right->next){
		left = left->next;
		right = right->next;
	}

	ListNode* newHead = left->next;
	left->next = nullptr;
	right->next = head;

	return newHead;
}

//189. Rotate Array
// 三次反转reverse, 实现简单，性能优良，且空间复杂度为O(1)
void MS::rotate(vector<int>& nums, int k){
	if(nums.empty()) return;
	k %= nums.size();
	std::reverse(nums.begin(), nums.end());
	std::reverse(nums.begin(), nums.begin() + k);
	std::reverse(nums.begin() + k, nums.end());
}

//82. Remove Duplicates from Sorted List II
ListNode* MS::deleteDuplicates2(ListNode* head) {
	ListNode* dummy = new ListNode(0, head);
	ListNode* p = dummy;
	while(p->next && p->next->next){
		if(p->next->val == p->next->next->val){
			int dupVal = p->next->val;
			while(p->next && p->next->val == dupVal){
				p->next = p->next->next;
			}
		}
		else{
			p = p->next;
		}
	}
	ListNode* newHead = dummy->next;
	delete dummy;
	return newHead;
}

//83. Remove Duplicates from Sorted List
ListNode* MS::deleteDuplicates(ListNode* head) {
	ListNode* dummy = new ListNode(0, head);
	ListNode* p = dummy;
	while(p->next && p->next->next){
		if(p->next->val == p->next->next->val){
			p->next = p->next->next;
		}
		else{
			p = p->next;
		}
	}
	ListNode* newHead = dummy->next;
	delete dummy;
	return newHead;
}

//141. Linked List Cycle
bool MS::hasCycle(ListNode *head){
	ListNode* fast = head;
	ListNode* slow = head;
	while(fast && fast->next){
		fast = fast->next->next;
		slow = slow->next;
		if(fast == slow){
			return true;
		}
	}

	return false;
}

//142.Linked List Cycle 2
// 1.环检测 2.找到环起点：从相遇点和链表头开始分别移动两个指针，每次移动一步。两指针最终会在环起点相遇。
ListNode* MS::detectCycle(ListNode *head){
	ListNode* fast = head;
	ListNode* slow = head;
	while(fast && fast->next){
		fast = fast->next->next;
		slow = slow->next;
		if(fast == slow){
			ListNode* entry = head;
			while(entry != fast){
				entry = entry->next;
				fast = fast->next;
			}
			return entry;
		}
	}

	return nullptr;
}

//148. Sort List
ListNode* MS::sortList(ListNode* head){
	if(!head || !head->next) return head;
	ListNode* slow = head;
	ListNode* fast = head->next;
	while(fast && fast->next){
		slow = slow->next;
		fast = fast->next->next;
	}
	ListNode* l2 = slow->next;
	slow->next = nullptr;

	ListNode* l1 = sortList(head);
	l2 = sortList(l2);

	return mergeTwoSorted(l1, l2);
}

ListNode* MS::mergeTwoSorted(ListNode* l1, ListNode* l2){
	if(!l1) return l2;
	if(!l2) return l1;
	ListNode* dummy = new ListNode();
	ListNode* p = dummy;
	while (l1 && l2){
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
	p->next = l1 ? l1 : l2;

	ListNode* newHead = dummy->next;
	delete dummy;
	return newHead;
}

//206. Reverse Linked List
ListNode* MS::reverseList(ListNode* head){
	ListNode* pre = nullptr;
	ListNode* cur = head;
	while (cur){
		ListNode* next = cur->next;
		cur->next = pre;
		pre = cur;
		cur = next;
	}

	return pre;
}

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
string MS::longestPalindrome(const std::string &s){
	if (s.empty()) return "";

	int start = 0;
	int longest = 0;
	for (int mid = 0; mid < s.size(); ++mid)
	{
		int len = findLongestPalindrome(s, mid, mid);
		if(len > longest){
			longest = len;
			start = mid - len / 2;
		}

		len = findLongestPalindrome(s, mid, mid + 1);
		if(len > longest){
			longest = len;
			start = mid - len / 2 + 1;
		}
	}

	return s.substr(start, longest);
}

int MS::findLongestPalindrome(const std::string &s, int i, int j){
	int len = 0;
	while(i >= 0 && j < s.size()){
		if(s.at(i) != s.at(j)){
			break;
		}
		len += i == j ? 1 : 2;
		i--;
		j++;
	}

	return len;
}

//22. Generate Parentheses
vector<string> MS::generateParenthesis(int n){
	int open = n;
	int close = n;
	string current;
	vector<string> result;
	back_tracking(open, close, current, result);
	return result;
}

void MS::back_tracking(int open, int close, string& current, vector<string>& result){
	// 如果没有剩余的括号可用，则将当前字符串加入结果中
	if(open == 0 && close == 0){
		result.push_back(current);
		return;
	}

	// 如果还有左括号可用，继续添加左括号
	if(open > 0){
		current.push_back('(');
		back_tracking(open - 1, close, current, result);
		current.pop_back();
	}

	// 如果右括号的数量大于左括号，可以添加右括号
	if(open < close){
		current.push_back(')');
		back_tracking(open, close - 1, current, result);
		current.pop_back();
	}
}

/**
     * LeetCode42.给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水
     * 思路
     *      1.用lMax[]和rMax[]分别存储第i个位置左侧和右侧最高值
     *      2.第i个位置的接水量 = min(lMax(i), rMax(i)) - height[i]
     *      3.时间复杂度 = 空间复杂度 = O(n)
     * */
int MS::trap(vector<int>& height){
	if(height.size() < 3) return 0;
	int n = height.size();
	int result = 0;
	vector<int> lMax(n);
	vector<int> rMax(n);
	lMax[0] = height[0];
	for (int i = 1; i < n; ++i)
	{
		lMax[i] = std::max(height[i], lMax[i - 1]);
	}
	rMax[n - 1] = height[n - 1];
	for (int i = n - 2; i >= 0; --i)
	{
		rMax[i] = std::max(height[i], rMax[i + 1]);
	}

	for (int i = 1; i < n; ++i)
	{
		result += std::min(lMax[i], rMax[i]) - height[i];
	}

	return result;
}

//55. JumpGame
bool MS::canJump(vector<int>& nums){
	int n = nums.size();
	vector<bool> dp(n, false);// dp[i] 表示是否可以到达位置 i
	dp[0] = true;
	for (int i = 1; i < n; ++i)
	{
		for (int j = 0; j < i; ++j)
		{
			if(dp[j] && nums[j] + j >= i)
			{
				dp[i] = true;
				break;
			}
		}
	}
	return dp[n - 1];
}

//45. JumpGame II
int MS::jump(vector<int>& nums){
	int n = nums.size();
	vector<int> dp(n);//dp[i] 表示到达位置 i 所需的最小跳跃次数。
	dp[0] = 0;
	// 从起点开始逐步计算每个位置所需的最小跳跃次数
	for(int i = 1; i < n; i++){
		dp[i] = INT_MAX;
		for(int j = 0; j < i; j++){
			if(j + nums[j] >= i){
				dp[i] = std::min(dp[i], dp[j] + 1);
			}
		}
	}

	return dp[n - 1];
}

//53. Maximum Subarray
int MS::maxSubArray(vector<int>& nums){
	int max = INT_MIN;
	int sum = 0;
	for(int n : nums){
		sum += n;
		max = std::max(max, sum);
		sum = std::max(sum, 0);
	}

	return max;
}


//198.House Robber
int MS::rob(vector<int>& nums){
	int n = nums.size();
	vector<int> dp(n + 1);//dp[n]表示偷前n个房屋的最大收益
	dp[0] = 0;
	dp[1] = nums[0];
	for (int i = 2; i <= n; ++i)
	{
		dp[i] = std::max(dp[i - 2] + nums[i - 1], dp[i - 1]);
	}

	return dp[n];
}


//213. House Robber II
int MS::robCircle(vector<int>& nums){
	if(nums.empty()) return 0;
	int n = nums.size();
	if(n == 1) return nums[0];
	vector<int> withFirst(nums.begin(), nums.end() - 1);
	int maxWithFirst = rob(withFirst);
	vector<int> withoutFirst(nums.begin() + 1, nums.end());
	int maxWithoutFirst = rob(withoutFirst);
	return std::max(maxWithFirst, maxWithoutFirst);
}

//377. Combination Sum IV
int MS::combinationSum4(vector<int>& nums, int target){
	//dp[target]表示和为target的所有组合数
	vector<int> dp(target + 1, 0);
	dp[0] = 1;
	for (int t = 1; t <= target; ++t)
	{
		for(int num : nums){
			if(t >= num){
				dp[t] += dp[t - num];
			}
		}
	}

	return dp[target];
}


//k数之和
int MS::kSum(vector<int>& nums, int k, int target){
	int n = nums.size();
	//dp[n][k][t] 表示从前n个数中取k个和为t的方案总数
	vector<vector<vector<int>>> dp(n + 1, vector<vector<int>>(k, vector<int>(target)));
	//dp[n][0][0] = 1
	for (int i = 0; i <= n; ++i)
	{
		dp[i][0][0] = 1;
	}
	for (int i = 1; i <= n; ++i)
	{
		for (int j = 1; j <= k && j <= i; ++j)
		{
			for (int t = 1; t <= target; ++t)
			{
				dp[i][j][t] = dp[i - 1][j][t];
				if(nums[i - 1] <= t){
					dp[i][j][t] += dp[i - 1][j - 1][t - nums[i - 1]];
				}
			}
		}
	}

	return dp[n][k][target];
}


/**
 * 62. Unique Paths 【坐标型coordinator type】
 * 有一个机器人的位于一个 m × n 个网格左上角。
 * 机器人每一时刻只能向下或者向右移动一步。机器人试图达到网格的右下角。
 * 问有多少条不同的路径？
 *
 * 1.确定状态
 *          走倒数第二步分为(m - 2, n - 1)、(m - 1, n - 2)
 *          子问题
 * 2.转移方程
 *          设f[i][j]表示走到(i, j)处的所有可能路径数
 * 3.初始条件和边界
 *          f[0][0] = 1;
 *          i = 0 || j = 0 -> f[i][j] = 1
 * 4.计算顺序
 *      遍历每行
 * */
int MS::uniquePaths(int m, int n){
	vector<vector<int>> dp(m, vector<int>(n));
	dp[0][0] = 1;
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			if(i == 0 || j == 0){
				dp[i][j] = 1;
				continue;
			}

			dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
		}
	}

	return dp[m - 1][n - 1];
}

//63. Unique Paths II
int MS::uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid){
	int m = obstacleGrid.size();
	int n = obstacleGrid.at(0).size();
	vector<vector<int>> dp(m, vector<int>(n));
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			if(obstacleGrid[i][j] == 1){
				dp[i][j] = 0;
				continue;
			}

			if(i == 0 && j == 0){
				dp[i][j] = 1;
				continue;
			}

			if(i > 0){
				dp[i][j] += dp[i - 1][j];
			}
			if(j > 0){
				dp[i][j] += dp[i][j - 1];
			}
		}
	}

	return dp[m - 1][n - 1];
}


/**
 * 64.Minimum Path Sum
 * 给定一个只含非负整数的m*n网格，找到一条从左上角到右下角的可以使数字和最小的路径。
 * 输入:  [[1,3,1],[1,5,1],[4,2,1]]
 * 	输出: 7
 *
 * 	样例解释：
 * 	路线为： 1 -> 3 -> 1 -> 1 -> 1。
 *
 * 1.确定状态
 *          走倒数第二步 前提下
 *          原问题 到右下角f[m - 1][n - 1]数字和最小的路径
 *          子问题 f[m - 2][n - 1] or f[m - 1][n - 2]数字和最小
 * 2.转移方程
 *          设f[i][j] 表示从左上角f[0][0]到f[m - 1][m - 1]的数字和最小路径
 * 3.初始条件和边界
 *          f[0][0] = grid[0][0];
 *
 * 4.计算顺序
 *      从左往右,从上至下
 * */

int MS::minPathSum(vector<vector<int>>& grid){
	int m, n = 0;
	if(grid.empty() || (m = grid.size()) == 0 || (n = grid.at(0).size()) == 0){
		return 0;
	}

	vector<vector<int>> dp(m, vector<int>(n));
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			if(i == 0 && j == 0){
				dp[i][j] = grid[i][j];
				continue;
			}

			dp[i][j] = INT_MAX;
			if(i > 0){
				dp[i][j] = std::min(dp[i][j], dp[i - 1][j]);
			}
			if(j > 0){
				dp[i][j] = std::min(dp[i][j], dp[i][j - 1]);
			}

			dp[i][j] += grid[i][j];
		}
	}

	return dp[m - 1][n - 1];
}

//115. Distinct Subsequences
int MS::numDistinct(string s, string t){
	if(s.empty()) return 0;
	int m = s.size();
	if(t.empty()) return 1;
	int n = t.size();
	//dp[i][j]表示s[0...i - 1]中含有t[0...j - 1]子序列的个数
	vector<vector<long>> dp(m + 1, vector<long>(n + 1, 0));
	////边界条件 dp[i][0] = 1
	for (int i = 0; i <= m; ++i)
	{
		dp[i][0] = 1;
	}
	const int MOD = 1e9 + 7;// 设置模数，防止溢出

	for (int i = 1; i <= m; ++i)
	{
		for (int j = 1; j <= n; ++j)
		{
			dp[i][j] = dp[i - 1][j];
			if(s.at(i - 1) == t.at(j - 1)){
				// dp[i][j] += dp[i - 1][j - 1];
				dp[i][j] = (dp[i][j] + dp[i - 1][j - 1]) % MOD;
			}
		}
	}

	return dp[m][n];
}

/**
  * linkCode392.打劫房屋
  * 假设你是一个专业的窃贼，准备沿着一条街打劫房屋。每个房子都存放着特定金额的钱。
  * 你面临的唯一约束条件是：相邻的房子装着相互联系的防盗系统，且 当相邻的两个房子同一天被打劫时，该系统会自动报警。
  *
  * 给定一个非负整数列表，表示每个房子中存放的钱， 算一算，如果今晚去打劫，在不触动报警装置的情况下, 你最多可以得到多少钱
  * 输入: [3, 8, 4]
  * 输出: 8
  * 解释: 仅仅打劫第二个房子.
  *
  * 输入: [5, 2, 1, 3]
  * 输出: 8
  * 解释: 抢第一个和最后一个房子
  *
  * 转移方程: f[i]表示偷前i栋房子的最大收益，最后一房子偷or不偷两种情况
  *          f[i] = max(f[i - 1], f[i - 2] + A[i])
  * */

/**
 * 房子成圈，收尾相连, 0 1 2 3 ... n-1
 * 分为两种情况: 1. 第0个房子不偷
 *             2. 第n-1个房子不偷
 */



/**
 * linkCode76. 最长上升子序列 【坐标型coordinator type】
 * 给定一个整数数组（下标从 0 到 n-1， n 表示整个数组的规模），请找出该数组中的最长上升连续子序列。
 * （最长上升连续子序列可以定义为从右到左或从左到右的序列。）
 *  输入: [4,2,4,5,3,7]
 * 	输出:  4
 *
 * 	解释:
 * 	LIS 是 [2,4,5,7]
 * 1.确定状态
 *          原问题 求以array[i]结尾的数组中的最长上升子序列
 *          子问题 求以array[i - 1]结尾的数组中的最长上升连续子序列
 * 2.转移方程
 *          设f[i]表示能到该位置的最长上升子序列的长度
 *          f[i] = f[j - 1] + 1 && j为枚举0 -> i中的值： i > j  && A[i] > A[j]
 * 3.初始条件和边界
 *          f[0] = 1;
 *
 * 4.计算顺序
 *      从左往右
 * */


/**
     * 对比：背包问题 line508
     * linkCode.89 k数之和
     * 输入：24，找出四个数字之和 = 24    输出：47种方案
     * */


/**
     * linkCode118. 不同的子序列, 给定字符串 S 和 T, 计算 S 的所有子序列中有多少个 T.
     * 输入: S = "rabbbit", T = "rabbit"
     * 输出: 3
     * 解释: 你可以删除 S 中的任意一个 'b', 所以一共有 3 种方式得到 T.
     *
     * 输入: S = "abcd", T = ""
     * 输出: 1
     * 解释: 只有删除 S 中的所有字符这一种方式得到 T
     * */


/**
     * linkCode397.最长上升连续子序列 【坐标型coordinator type】
     * 给定一个整数数组（下标从 0 到 n-1， n 表示整个数组的规模），请找出该数组中的最长上升连续子序列。
     * （最长上升连续子序列可以定义为从右到左或从左到右的序列。）
     * 输入：[5, 1, 2, 3, 4]
     * 输出：4
     * 1.确定状态
     *          走倒数第二步 f[i] = f[i - 1] + 1 在a[i] > a[i - 1] && i > 0前提下
     *          原问题 求以array[i]结尾的数组中的最长上升连续子序列
     *          子问题 求以array[i - 1]结尾的数组中的最长上升连续子序列
     * 2.转移方程
     *          设f[i]表示能到该位置的最长连续子序列的长度
     * 3.初始条件和边界
     *          f[0] = 1;
     *
     * 4.计算顺序
     *      从左往右
     * */


/**
 * ‼️背包问题中，dp数组大小和总承重有关系‼️
 * linkCode92. 背包问题
 * 在n个物品中挑选若干物品装入背包，最多能装多满？假设背包的大小为m，每个物品的大小为A[i]
 * m:最大承重
 *
 * 确定状态：dp[i][w] 表示前i个物品是否能拼出重量w
 * 转移方程：dp[i][w] = dp[i - 1][w] || dp[i - 1][w - A[i]]
 *
 * */




//endregion



//region Sort

//region QuickSort 最优、平均:O(nlog(n))   最差:O(n平方)
void MS::quick_sort(vector<int>& array){
	quick_sort(array, 0, (int)array.size() - 1);
}

void MS::quick_sort(vector<int>& array, int start, int end){
	if(start >= end){
		return;
	}

	// 使用三数取中法选择基准值
	// 对于某些特定输入（例如，数组已经有序），这可能导致极端不平衡的分区，使时间复杂度退化为 O(n平方)
	int pivot_index = start + (end - start) / 2;
	if ((array[start] > array[pivot_index]) != (array[start] > array[end])) {
		pivot_index = start;
	} else if ((array[end] > array[start]) != (array[end] > array[pivot_index])) {
		pivot_index = end;
	}
	swap(array[start], array[pivot_index]);

	int pivot = internal_quick_sort(array, start, end);

	quick_sort(array, start, pivot - 1);
	quick_sort(array, pivot + 1, end);
}

int MS::internal_quick_sort(vector<int>& array, int start, int end){
	int left = start;
	int right = end;
	int temp = array[left];
	while(left < right){
		while(left < right && array[right] >= temp){
			right--;
		}
		array[left] = array[right];
		while(left < right && array[left] <= temp){
			left++;
		}
		array[right] = array[left];
	}

	array[left] = temp;
	return left;
}

//endregion


//region Insert Sort

void MS::insert_sort(vector<int>& array){
	int size = array.size();
	for(int i = 1; i < size; i++){
		int key = array[i];
		int j = i - 1;
		// 在已排序部分中找到插入位置
		while(j >= 0 && array[j] > key){
			array[j + 1] = array[j];
			j--;
		}

		array[j + 1] = key;
	}
}

//endregion

void MS::merge_sort(vector<int>& array){
	vector<int> temp(array.size());
	merge_sort(array, 0, array.size() - 1, temp);
}

void MS::merge_sort(vector<int>& array, int start, int end, vector<int>& temp){
	if(start >= end){
		return;
	}
	int mid = start + (end - start) / 2;
	merge_sort(array, start, mid, temp);
	merge_sort(array, mid + 1, end, temp);
	internal_merge_sort(array, start, mid, end, temp);
}

void MS::internal_merge_sort(vector<int>& array, int start, int mid, int end, vector<int>& temp){
	int i = start;
	int j = mid + 1;
	int k = start;
	while(i <= mid && j <= end){
		temp[k++] = array[i] < array[j] ? array[i++] : array[j++];
	}

	while(i <= mid){
		temp[k++] = array[i++];
	}

	while(j <= end){
		temp[k++] = array[j++];
	}

	//这里存在一定的性能开销，可以直接用循环代替 std::copy。因为在小范围数据复制时，手动循环比 std::copy 通常更高效。
	//std::copy(temp.begin() + start, temp.begin() + end + 1, array.begin() + start);//

	// 可以直接在 temp 上完成排序，递归结束后统一复制回原数组。或者将 array 和 temp 交换使用，避免频繁的拷贝。
	for (int t = start; t <= end; ++t)
	{
		array[t] = temp[t];
	}
}

void MS::heap_sort(vector<int>& array){
	if(array.empty()) return;
	int length = array.size();
	for (int i = length / 2 - 1; i >= 0; --i)
	{
		heap_adjust(array,i, length);
	}

	for (int i = length - 1; i >= 0; --i)
	{
		std::swap(array[0], array[i]);
		heap_adjust(array, 0, i);
	}
}

void MS::heap_adjust(vector<int>& array, int index, int length){
	int largest = index;
	int left = index * 2 + 1;
	int right = left + 1;
	if(left < length && array[left] > array[largest]){
		largest = left;
	}

	if(right < length && array[right] > array[largest]){
		largest = right;
	}

	if(largest != index){
		std::swap(array[largest], array[index]);
		heap_adjust(array, largest, length);
	}
}

void MS::heap_adjust_non_recursive(vector<int>& array, int index, int length){
	while (true){
		int largest = index;
		int left = index * 2 + 1;
		int right = left + 1;
		if(left < length && array[left] > array[largest]){
			largest = left;
		}

		if(right < length && array[right] > array[largest]){
			largest = right;
		}

		if(largest != index){
			std::swap(array[largest], array[index]);
			index = largest;
		}
		else{
			break;//// 当前节点已满足堆性质
		}
	}
}

//347. Top K Frequent Elements
//1.统计频率
//2.构建小根堆
//3.遍历频率表，将元素加入堆中，维护堆的大小为 k
//4.提取堆中元素，放入结果数组
vector<int> MS::topKFrequent(vector<int>& nums, int k){
	unordered_map<int, int> freq_map;
	for(int n : nums){
		freq_map[n]++;
	}

	auto compare = [](pair<int, int> a, pair<int, int> b){
		return a.second > b.second;
	};
	priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(compare)> minHeap(compare);

	for(auto& pair : freq_map){
		//避免显式构造 pair<int, int> 的中间变量，直接将 num 和 freq 原地构造成 pair<int, int>
		minHeap.emplace(pair);
		if(minHeap.size() > k){
			minHeap.pop();
		}
	}

	vector<int> result;
	while (!minHeap.empty()){
		result.push_back(minHeap.top().first);//获取频率高对应的数字
		minHeap.pop();
	}

	reverse(result.begin(), result.end());
	return result;
}

//endregion