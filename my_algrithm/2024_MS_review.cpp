//
// Created by xiangxx on 2024/10/15.
//

#include "2024_MS_review.h"

#include <vector>
#include <stack>
#include <queue>
#include <unordered_map>
#include <unordered_set>
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

/*
 * 3. Longest Substring Without Repeating Characters
 * Time: O(n)
 * Space: O(n)
 * */
int MS::lengthOfLongestSubstring(string s){
	int n = s.size();
	int longest = 0;
	int left = 0;
	unordered_map<char, int> char_map;// 哈希表用于记录字符的最近位置
	for (int right = 0; right < n; ++right)
	{
		// 如果当前字符在哈希表中已存在，并且它的位置大于等于左边界
		if(char_map.count(s[right]) && char_map[s[right]] >= left){
			left = char_map[s[right]] + 1;
		}
		char_map[s[right]] = right;
		longest = std::max(longest, right - left + 1);
	}

	return longest;
}

//7. Reverse Integer
int MS::reverseInteger(int x){
	int res = 0;
	while(x != 0){
		if(res > INT_MAX / 10 || (res == INT_MAX / 10 && x % 10 > INT_MAX % 10)){
			return 0;
		}

		if(res < INT_MIN / 10 || (res == INT_MIN / 10 && x % 10 < INT_MIN % 10))
		{
			return 0;
		}
		res = res * 10 + x % 10;
		x /= 10;
	}

	return res;
}

/*
 * 15.3Sum
 * 思路：遍历并固定第i个数，双指针寻找另外两个数 使得三数之和为0， 注意去重
 * Time:O(n^2),其中 n 为数组的长度。排序 O(n log n)，双指针遍历过程 O(n^2)
 * Space: O(k + log n)，结果三元组的数量 + 排序所需的栈空间
 * */
vector<vector<int>> MS::threeSum(vector<int>& nums){
	if(nums.empty()) return {};
	std::sort(nums.begin(), nums.end());
	vector<vector<int>> result;
	int n = nums.size();
	for (int i = 0; i < n; ++i)
	{
		if(i > 0 && nums[i] == nums[i - 1]){
			continue;
		}

		int left = i + 1;
		int right = n - 1;
		while(left < right){
			int sum = nums[i] + nums[left] + nums[right];
			if(sum == 0){
				result.push_back({nums[i], nums[left], nums[right]});
				left++;
				right--;

				while(left < right && nums[left] == nums[left - 1]){
					left++;
				}
				while(left < right && nums[right] == nums[right + 1])
				{
					right--;
				}
			}
			else if(sum > 0){
				right--;
			}
			else{
				left++;
			}
		}
	}

	return result;
}

//31.Next permutation
void MS::nextPermutation(vector<int>& nums){
	int n = nums.size();
	int i = n - 2;
	while(i >= 0 && nums[i] >= nums[i + 1]){ //找到满足 nums[i] < nums[i + 1] 的最大索引 i
		i--;
	}
	if(i >= 0){
		int j = n - 1;
		while(j >= 0 && nums[j] <= nums[i]){ //从后往前找到第一个大于 nums[i] 的元素
			j--;
		}
		swap(nums[i], nums[j]);
	}

	std::reverse(nums.begin() + i + 1, nums.end());
}

/*
 * 39. Combination Sum
 * 给定一个无重复元素的数组 candidates 和一个目标数 target，找出 candidates 中所有可以使数字和为 target 的组合。
 * 同一个数字在组合中可以使用多次
 * Time: 取决于回溯深度以及分支情况，最坏情况接近 O(n^(target/min_num))，实际常用场景下不严格这样测
 * Space: O(target)（回溯深度）或 O(n)（取决于输入规模），主要是递归深度和暂存路径
 *
 * */
vector<vector<int>> MS::combinationSum(vector<int>& candidates, int target){
	vector<vector<int>> result;
	if(candidates.empty() || target == 0) return result;
	std::sort(candidates.begin(), candidates.end());
	vector<int> subset;
	combinationSumHelper(candidates, target, 0, subset, result);
	return result;
}

void MS::combinationSumHelper(vector<int>& candidates, int remained, int start, vector<int>& subset, vector<vector<int>>& result){
	if(remained == 0){
		result.push_back(subset);
		return;
	}

	for (int i = start; i < candidates.size(); ++i)
	{
		if(remained < candidates[i]){
			break;
		}
		subset.push_back(candidates[i]);
		// 同一元素可重复使用，故下一次调用的 start 依然是 i
		combinationSumHelper(candidates, remained - candidates[i], i, subset, result);
		subset.pop_back();
	}
}

//46.Permutations
/*
 * 时间复杂度:O(n·n!)
 * 空间复杂度：O(n·n!)：因为
 * */
vector<vector<int>> MS::permute(vector<int>& nums) {
	if(nums.empty()){
		return {};
	}

	vector<vector<int>> result;
	vector<bool> visited(nums.size(), false);
	vector<int> temp;
	permute_dfs(nums, visited, temp, result);
	return result;
}

void MS::permute_dfs(vector<int>& nums, vector<bool>& visited, vector<int>& temp, vector<vector<int>>& result){
	if(nums.size() == temp.size()){
		result.push_back(temp);
		return;
	}

	for(int i = 0; i < nums.size(); i++){
		if(visited[i]){
			continue;
		}

		visited[i] = true;
		temp.push_back(nums[i]);
		permute_dfs(nums, visited, temp, result);
		temp.pop_back();
		visited[i] = false;
	}
}

vector<vector<int>> MS::permuteUnique(vector<int>& nums) {
	if(nums.empty()){
		return {};
	}
	std::sort(nums.begin(), nums.end());

	vector<vector<int>> result;
	vector<bool> visited(nums.size(), false);
	vector<int> temp;
	permuteUnique_dfs(nums, visited, temp, result);
	return result;
}

void MS::permuteUnique_dfs(vector<int>& nums, vector<bool>& visited, vector<int>& temp, vector<vector<int>>& result){
	if(nums.size() == temp.size()){
		result.push_back(temp);
		return;
	}

	for(int i = 0; i < nums.size(); i++){
		if(i > 0 && nums[i] == nums[i - 1] && !visited[i - 1]){
			continue;
		}

		if(visited[i]){
			continue;
		}

		visited[i] = true;
		temp.push_back(nums[i]);
		permuteUnique_dfs(nums, visited, temp, result);
		temp.pop_back();
		visited[i] = false;
	}
}

/*
 * 48.Rotate Image
 * 给定一个 n x n 的二维矩阵 matrix，表示一幅图像，请将图像原地旋转 90 度（顺时针）
 * 1 2 3	  7 4 1
 * 4 5 6  ->  8 5 2
 * 7 8 9	  9 6 3
 * 思路：先交换 swap(matrix[i][j], matrix[j][i]), 再对每一行reverse
 * Time: O(n^2),访问矩阵的每个元素
 * Space: O(1)
 *
 * */
void MS::rotate(vector<vector<int>>& matrix){
	int m = matrix.size();
	int n = matrix[0].size();
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < i; ++j)
		{
			swap(matrix[i][j], matrix[j][i]);
		}
	}

	for(int i = 0; i < m; i++){
		std::reverse(matrix[i].begin(), matrix[i].end());
	}
}

/*
 * 49. Group Anagrams 由相同字符但排列顺序不同的字符串
 * 思路：对每个字符串排序，作为哈希表的键值key，value为vector<string>存储相同key的字符串
 * Time: O(n * k log k)，其中 n 为字符串数目，k 为字符串平均长度。对每个字符串排序需要 O(k log k)，总共 n 个字符串
 * Space: O(nk)，哈希表中存储所有字符串副本
 *
 * C++ 中，unordered_map 的行为是这样的：当通过键访问一个不存在的元素时，会隐式地初始化一个默认值
 * */
vector<vector<string>> MS::groupAnagrams(vector<string>& strs){
	unordered_map<string, vector<string>> strMap;
	for(auto& str : strs){
		string sortedStr = str;
		std::sort(sortedStr.begin(), sortedStr.end());
		strMap[sortedStr].push_back(str);//隐式初始化
	}
	vector<vector<string>> result;
	result.reserve(strMap.size());
	for (const auto& item: strMap){
		result.push_back(item.second);
	}

	return result;
}

//56.Merge Intervals 合并区间
/*
 * Time：O(nlogn)
 * Space：O(n)
 * */
vector<vector<int>> MS::mergeIntervals(vector<vector<int>>& intervals){
	if(intervals.empty()){
		return intervals;
	}

	std::sort(intervals.begin(),
			intervals.end(),
			[](vector<int>& a, vector<int>& b){
				return a[0] < b[0];
			});

	vector<vector<int>> result;
	result.push_back(intervals[0]);
	int n = intervals.size();
	for (int i = 1; i < n; ++i)
	{
		vector<int>& current = intervals[i];//❗️必须返回的是vector<int>&
		vector<int>& last = result.back();
		if(current.front() <= last.back()){
			// 更新合并区间的结束位置
			last.back() = std::max(last.back(), current.back());
		}
		else{
			//没有重叠，添加分区
			result.push_back(current);
		}
	}

	return result;
}

//70.Climbing Stairs
// Time：O(n)
// Space:O(1)
int MS::climbStairs(int n){
	if (n <= 2) return n; // 边界情况

	int prev1 = 2; // 表示 dp[i-1]
	int prev2 = 1; // 表示 dp[i-2]

	for (int i = 3; i <= n; ++i) {
		int current = prev1 + prev2; // 当前的 dp[i]
		prev2 = prev1; // 更新 dp[i-2]
		prev1 = current; // 更新 dp[i-1]
	}

	return prev1;
}

/*
 * 78. Subsets
 * 给定一个不含重复元素的整数数组 nums，返回其所有可能的子集（幂集）
 * Time:O(n × 2^n)
 * 		每个子集的构造时间为 O(n)
 * 		每次迭代将现有的子集扩展，生成2^n个子集
 * Space:O(n)
 * */
vector<vector<int>> MS::subsets(vector<int>& nums) {
	if(nums.empty()) return {};
	//std::sort(nums.begin(), nums.end());
	vector<vector<int>> result;
	vector<int> temp;
	subsetsDFS(nums, 0, temp, result);
	return result;
}

void MS::subsetsDFS(vector<int>& nums, int index, vector<int>& temp, vector<vector<int>>& result){
	result.push_back(temp);

	for(int i = index; i < nums.size(); i++){
		temp.push_back(nums[i]);
		subsetsDFS(nums, i + 1, temp, result);
		temp.pop_back();
	}
}

/*
 * 90.Subsets 2
 * Time:O(n × 2^n)
 * Space:O(n × 2^n)
 *
 * */
vector<vector<int>> MS::subsetsWithDup(vector<int>& nums){
	if(nums.empty()) return {};
	std::sort(nums.begin(), nums.end());
	vector<vector<int>> result;
	vector<int> temp;
	subsetsWithDup(nums, 0, temp, result);
	return result;
}

void MS::subsetsWithDup(vector<int>& nums, int index, vector<int>& temp, vector<vector<int>>& result){
	result.push_back(temp);

	for(int i = index; i < nums.size(); i++){
		if(i > index && nums[i] == nums[i - 1]){
			continue;
		}
		temp.push_back(nums[i]);
		subsetsWithDup(nums, i + 1, temp, result);
		temp.pop_back();
	}
}

/*
 * 215.Kth Largest Element in an Array
 * Time:平均 O(n)，最坏 O(n^2)
 * Space:O(log n)，递归栈空间（平均情况）
 * 快速选择：更高效，尤其适合静态数组
 * */
int MS::findKthLargest(vector<int>& nums, int k){
	return quickSortHelper(nums, 0, nums.size() - 1, nums.size() - k);
}

/*
 * 215.Kth Largest Element in an Array
 * Time:总时间复杂度为 O(n log k),插入堆和删除堆顶的时间复杂度为 O(log k)，遍历数组需要 O(n)
 * Space:堆的大小为 k，空间复杂度为 O(k)
 *	优先队列：更适合用于流式数据，保持动态更新
 * */
int MS::findKthLargestMinHeap(vector<int>& nums, int k){
	priority_queue<int, vector<int>, greater<int>> minHeap;
	for(int n : nums){
		minHeap.push(n);
		if(minHeap.size() > k){
			minHeap.pop();
		}
	}

	return minHeap.top();
}

int MS::quickSortHelper(vector<int>& nums, int start, int end, int index){
	int pivot = internal_quick_sort(nums, start, end);
	if(pivot == index){
		return nums[pivot];
	}
	else if(pivot > index){
		return quickSortHelper(nums, start, pivot - 1, index);
	}
	else{
		return quickSortHelper(nums, pivot + 1, end, index);
	}
}

/*
 * 238. Product of Array Except Self
 * 给定一个整数数组 nums，返回一个数组 answer，其中 answer[i] 等于 nums 除下标 i 之外的所有元素的乘积。
 * 要求在 O(n) 时间复杂度和不使用除法的情况下完成。
 * 思路：分别设置left数组、right数组
 * 		left[i]表示nums[0, i - 1]所有数乘积
 * 		right[i]表示nums[i + 1, n - 1]所有数乘积
 * 		最终结果 answers[i] = left[i] * right[i]
 * Time: O(n) 遍历两次
 * Space: O(1)
 *
 * */
vector<int> MS::productExceptSelf(vector<int>& nums){
	int n = nums.size();
	vector<int> answers(n, 1);

	// answer[i] 存储的是 nums[0..i-1] 的乘积
	for (int i = 1; i < n; ++i)
	{
		answers[i] = answers[i - 1] * nums[i - 1];
	}

	// 通过一个变量记录右侧乘积
	int rightProduct = 1;
	for(int i = n - 1; i >= 0; i--){
		answers[i] = answers[i] * rightProduct;
		rightProduct *= nums[i];
	}

	return answers;
}

//746. Min Cost Climbing Stairs
int MS::minCostClimbingStairs(vector<int>& cost){
	if(cost.empty()) return 0;
	int n = cost.size();
	/*
	 * Time:O(n) Space:O(1)
	 * */
	if(n == 0) return 0;
	if(n == 1) return cost[0];
	int prev1 = 0;
	int prev2 = 0;
	for(int i = 2; i <= n; i++){
		int cur = std::min(prev1 + cost[i - 1], prev2 + cost[i - 2]);
		prev2 = prev1;
		prev1 = cur;
	}

	return prev1;

	/*
	 * dp实现，Time: O(2xn) Space:O(n)
	 * */
//	int sum = 0;
//	vector<int> dp(n + 1);//dp[i]表示cost[i]处的最小cost, return min(dp[n - 1], dp[n - 2]);
//	dp[0] = 0;
//	dp[1] = cost[0];
//	for (int i = 2; i <= n; ++i)
//	{
//		dp[i] = INT_MAX;
//		for (int j = 1; j <= 2; ++j)
//		{
//			dp[i] = std::min(dp[i], dp[i - j]);
//		}
//		dp[i] += cost[i - 1];
//	}
//
//	return std::min(dp[n], dp[n - 1]);
}

//322. Coin Change
/*
 * 状态定义：dp[i]表示凑成金额i的最少硬币数
 * 状态转移方程: dp[i] = min(dp[i], dp[i - coin] + 1) | i >= coin 且 dp[i - coin]不为INT_MAX 时
 * 初始化条件：dp[0] = 0,其他值初始化为正无穷
 * 返回结果: INT_MAX ? -1 : dp[amount]
 *
 * Time: O(amount * coin.size()) Space: O(amount)
 * */
int MS::coinChange(vector<int>& coins, int amount){
	vector<int> dp(amount + 1, INT_MAX);
	dp[0] = 0;
	for (int i = 1; i <= amount; ++i)
	{
		for (const auto& coin: coins){
			if(i >= coin && dp[i - coin] != INT_MAX){
				dp[i] = std::min(dp[i], dp[i - coin] + 1);
			}
		}
	}

	return dp[amount] == INT_MAX ? -1 : dp[amount];
}

//344.Reverse String
void MS::reverseString(vector<char>& s){
	int left = 0;
	int right = s.size() - 1;
	while(left < right){
		swap(s[left], s[right]);
		left++;
		right--;
	}
}

//905. Sort Array By Parity (偶 + 奇 -> 双指针)
vector<int> MS::sortArrayByParity(vector<int>& nums){
	int left = 0;
	int right = nums.size() - 1;
	while(left < right){
		if(nums[left] % 2 > nums[right] % 2){
			swap(nums[left], nums[right]);
		}

		if(nums[left] % 2 == 0) left++;
		if(nums[right] % 2 == 1) right--;
	}

	return nums;
}

//region Stack and Queue

//20. Valid Parentheses 有效括号
bool MS::isValidParentheses(string s){
	if(s.empty()) return true;
	stack<char> sk;
	for(auto& ch : s){
		if(ch == ')' || ch == ']' || ch == '}'){
			if(sk.empty()
			   || (ch == ')' && sk.top() != '(')
			   || (ch == ']' && sk.top() != '[')
			   || (ch == '}' && sk.top() != '{')){
				return false;
			}
			sk.pop();
		}
		else{
			sk.push(ch);
		}
	}

	return sk.empty();
}

//endregion


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

/*
 * 95. Unique Binary Search Trees II
 *
 * Time: 生成所有可能的二叉搜索树需要遍历所有可能的组合，时间复杂度接近 O(4^n / sqrt(n))
 * Space: 树的递归生成和存储需要 O(4^n / sqrt(n))
 * */

vector<TreeNode*> MS::generateTrees(int n){
	if(n == 0) return {};
	return generateTreesHelper(1, n);
}

vector<TreeNode*> MS::generateTreesHelper(int start, int end){
	if(start > end) return { nullptr };
	vector<TreeNode*> result;
	for(int i = start; i <= end; i++){
		vector<TreeNode*> leftTrees = generateTreesHelper(start, i - 1);
		vector<TreeNode*> rightTrees = generateTreesHelper(i + 1, end);
		for (auto l: leftTrees)
		{
			for (auto r: rightTrees)
			{
				TreeNode* root = new TreeNode(i);
				root->left = l;
				root->right = r;
				result.push_back(root);
			}
		}
	}

	return result;
}

/*
 * 96: Unique Binary Search Trees
 * 定义 dp[i] 表示 i 个节点能构成的不同二叉搜索树的个数
 * 对于每个数 j（1 ≤ j ≤ i），将其作为根节点：
		左子树有 j-1 个节点，右子树有 i-j 个节点。
		总数为 dp[j-1] * dp[i-j]。
 * */
int MS::numTrees(int n){
	vector<int> dp(n + 1);//dp[i] 表示 i 个节点能构成的不同二叉搜索树的个数
	dp[0] = 1;
	for(int i = 1; i <= n; i++){
		for (int j = 1; j <= i; ++j)
		{
			dp[i] += dp[j - 1] * dp[i - j];
		}
	}

	return dp[n];
}

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

/*
 * 103.Binary Tree Zigzag Level Order Traversal
 * 在二叉树的层序遍历基础上，要求交替方向（从左到右，再从右到左）打印每层节点
 * Time:
 * Space:
 * */
vector<vector<int>> MS::zigzagLevelOrder(TreeNode* root){
	if(!root) return {};
	queue<TreeNode*> q;
	q.push(root);
	vector<vector<int>> result;
	bool left_to_right = true;
	while(!q.empty()){
		int size = q.size();
		vector<int> cur_level(size);
		for (int i = 0; i < size; ++i)
		{
			TreeNode* node = q.front();
			int index = left_to_right ? i : size - i - 1;
			cur_level[index] = node->val;
			if(node->left) q.push(node->left);
			if(node->right) q.push(node->right);
			q.pop();
		}

		left_to_right = !left_to_right;
		result.push_back(cur_level);
	}

	return result;
}


/*
 * 105.Construct Binary Tree from Preorder and Inorder Traversal
 * Time:O(n)
 * Space:
 * 	平均情况下（平衡二叉树），递归深度为 O(log n)
 * 	最坏情况下（树退化为链表），递归深度为 O(n)。
 *  存储中序遍历的 n 个元素，空间复杂度为 O(n)
 * */
TreeNode* MS::buildTree(vector<int>& preorder, vector<int>& inorder) {
	unordered_map<int, int> inOrderMap;
	for(int i = 0; i < inorder.size(); i++){
		inOrderMap[inorder[i]] = i;
	}
	int preIndex = 0;
	return buildTreeHelper(preorder, inorder, inOrderMap, preIndex, 0, inorder.size() - 1);
}

TreeNode* MS::buildTreeHelper(vector<int>& preorder, vector<int>& inorder, unordered_map<int, int>& inOrderMap, int& preIndex, int inLeft, int inRight){
	if(inLeft > inRight){
		return nullptr;
	}
	int val = preorder[preIndex];
	preIndex++;
	TreeNode* root = new TreeNode(val);
	int inIndex = inOrderMap[val];
	root->left = buildTreeHelper(preorder, inorder, inOrderMap, preIndex, inLeft, inIndex - 1);
	root->right = buildTreeHelper(preorder, inorder, inOrderMap, preIndex, inIndex + 1, inRight);
	return root;
}

/*
 * 109. Convert Sorted List to Binary Search Tree
 *	 Given the head of a singly linked list where elements are sorted in ascending order, convert it to a
 * 	height-balanced binary search tree.

 * Time: O(n)
 * Space: O(n)
 * */
TreeNode* MS::sortedListToBST(ListNode* head){
	if(!head) return nullptr;
	vector<int> nums = ListNodeToArray(head);
	return buildBST(nums, 0, nums.size() - 1);
}

vector<int> MS::ListNodeToArray(ListNode* head){
	vector<int> result;
	ListNode* cur = head;
	while(cur){
		result.push_back(cur->val);
		cur = cur->next;
	}

	return result;
}

TreeNode* MS::buildBST(vector<int>& nums, int start, int end){
	if(start > end){
		return nullptr;
	}
	int mid = start + (end - start) / 2;
	TreeNode* root = new TreeNode(nums[mid]);

	root->left = buildBST(nums, start, mid - 1);
	root->right = buildBST(nums, mid + 1, end);

	return root;
}

/*
 * 110. Balanced Binary Tree
 * Time:O(n)：每个节点只访问一次
 * Space:
 * 	O(h)：递归调用栈的深度为树的高度 h;
 * 	最坏情况下（链表状树）：O(n)
 * 	平衡二叉树：O(log n)
 * */
bool MS::isBalanced(TreeNode* root){
	return isBalancedHelper(root) != NOT_BALANCED;
}

int MS::isBalancedHelper(TreeNode* root){
	if (!root) return 0;

	int left = isBalancedHelper(root->left);
	int right = isBalancedHelper(root->right);

	if (left == NOT_BALANCED || right == NOT_BALANCED || abs(left - right) > 1) {
		return NOT_BALANCED;
	}

	return max(left, right) + 1;
}

/*
 * 112.Path Sum
 * Time:O(n)：每个节点只访问一次
 * Space:
 * 		最坏情况下（完全不平衡树）：O(n)
 * 		最好情况下（完全平衡树）：O(log n)
 * */
bool MS::hasPathSum(TreeNode* root, int targetSum){
	if(!root) return false;
	if(!root->left && !root->right && root->val == targetSum) return true;
	return hasPathSum(root->left, targetSum - root->val) ||
		   hasPathSum(root->right, targetSum - root->val);
}

/*
 * 113. Path Sum II
 * 思路：dfs + 回溯
 * Time: O(n)
 * Space:O(h + n)
 * */
vector<vector<int>> MS::pathSum(TreeNode* root, int targetSum){
	vector<vector<int>> result;
	vector<int> temp;
	pathSumDFS(root, targetSum, temp, result);
	return result;
}

void MS::pathSumDFS(TreeNode* root, int targetSum, vector<int>& path, vector<vector<int>>& result){
	if(!root){
		return;
	}
	path.push_back(root->val);

	if(!root->left && !root->right && root->val == targetSum){
		result.push_back(path);
	}

	pathSumDFS(root->left, targetSum - root->val, path, result);
	pathSumDFS(root->right, targetSum - root->val, path, result);
	path.pop_back();
}


/*
 * 114. Flatten Binary Tree to Linked List
 * 一棵二叉树，将它展开为一个类似单链表的结构。要求展开后单链表仍然使用 right 指针作为下一个节点指向，left 指针置为 nullptr
 * Time:O(n)
 * Space:最坏O(h), h为树的高度
 * */
void MS::flatten(TreeNode* root){
	if(!root) return;

	flatten(root->left);
	flatten(root->right);

	TreeNode* temp = root->right;
	root->right = root->left;
	root->left = nullptr;

	TreeNode* cur = root;
	while(cur->right){
		cur = cur->right;
	}
	cur->right = temp;
}

//144. Binary Tree Preorder Traversal
vector<int> MS::preorderTraversal(TreeNode* root){
	if(root == nullptr) return {};
	vector<int> result;
	stack<TreeNode*> sk;
	TreeNode* p = root;
	while(p || !sk.empty()){
		if(p != nullptr){
			sk.push(p);
			result.push_back(p->val);
			p = p->left;
		}
		else{
			p = sk.top();
			sk.pop();
			p = p->right;
		}
	}

	return result;
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

/*
 * 236. Lowest Common Ancestor of a Binary Tree
 * Time:   O(n)，n 为节点数，DFS 需遍历整棵树一次
 * Space:  O(n)，最坏情况下（树退化成链表）递归深度可达n
 * */
TreeNode* MS::lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q){
	if(!root || root == p || root == q) return root;
	TreeNode* left = lowestCommonAncestor(root->left, p, q);
	TreeNode* right = lowestCommonAncestor(root->right, p, q);
	if(left && right){
		return root;
	}

	return left ? left : right;
}

/*
	 * 285. 二叉搜索树中的中序后继
	 * 给定一棵二叉搜索树和其中的一个节点 p，找到该节点在树中的中序后继。
	 * 如果节点没有中序后继，请返回 null。 节点 p 的后继是值比 p.val 大的节点中键值最小的节点
	 * */
TreeNode* MS::inorderSuccessor(TreeNode* root, TreeNode* p){
	TreeNode* successor = nullptr;
	while (root){
		if(p->val < root->val){
			successor = root;
			root = root->left;
		}
		else{
			root = root->right;
		}
	}

	return successor;
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

/*
 * 82. Remove Duplicates from Sorted List II
 * Time:O(n)
 * Space:O(1)
 * */
ListNode* MS::deleteDuplicates2(ListNode* head) {
	ListNode dummy;
	ListNode* p = &dummy;
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

	return dummy.next;
}

/*
 * 83. Remove Duplicates from Sorted List
 * Time:O(n)
 * Space:O(1)
 * */
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

/*
 * 92. Reverse Linked List II
 * Time
  	查找起点和终点的循环各为 O(n)。
	反转子链表部分的复杂度为 O(right - left + 1)。
	总时间复杂度为 O(n)
 * Space:O(1)
 * */
ListNode* MS::reverseBetween(ListNode* head, int left, int right){
	if (!head || left == right) return head;

	ListNode dummy;
	dummy.next = head;
	ListNode* slow = &dummy;

	for(int i = 1; i < left; i++){
		slow = slow->next;
	}

	ListNode* fast = slow;
	for(int i = 0; i <= right - left + 1; i++){
		fast = fast->next;
	}

	ListNode* tail = slow->next;
	slow->next = reverseHelper(tail, fast);
	tail->next = fast;

	ListNode* res = dummy.next;
	return res;
}

ListNode* MS::reverseHelper(ListNode* head, ListNode* target){
	ListNode* cur = head;
	ListNode* pre = nullptr;
	while(cur != target){
		ListNode* next = cur->next;
		cur->next = pre;
		pre = cur;
		cur = next;
	}

	return pre;
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

//143. Reorder List
void MS::reorderList(ListNode* head){
	if(!head || !head->next) return;
	ListNode* p = head;
	ListNode* q = reverseList(partitionList(p));
	while(p && q){
		ListNode* pNext = p->next;
		p->next = q;
		ListNode* qNext = q->next;

		p = pNext;
		if(p){
			q->next = p;
		}

		q = qNext;
	}
}

ListNode* MS::partitionList(ListNode* head){
	if(!head || !head->next) return head;
	ListNode* slow = head;
	ListNode* fast = head->next->next;
	while(fast && fast->next){
		slow = slow->next;
		fast = fast->next->next;
	}
	ListNode* next = slow->next;
	slow->next = nullptr;

	return next;
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
/*
 * 5.Longest Palindromic Substring
 *
 * Time:O(n^2)
 * Space: O(1)
 * */
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
	vector<string> result;
	back_tracking("", n, n, result);
	return result;
}

void MS::back_tracking(string cur, int left, int right, vector<string>& result){
	if(left == 0 && right == 0){
		result.push_back(cur);
		return;
	}
	//如果还有左括号可以用
	if(left > 0){
		back_tracking(cur + '(', left - 1, right, result);
	}
	//只有当右括号数量大于左括号数量时，才能使用右括号
	if(left < right){
		back_tracking(cur + ')', left, right - 1, result);
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

//55. JumpGame O(n2)
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

bool MS::canJump_Greed(vector<int>& nums){
	int farthest = 0;// 当前能到达的最远位置
	int n = nums.size();
	for (int i = 0; i < n; ++i)
	{
		if(i > farthest){
			return false;// 如果当前索引超出了能到达的范围
		}

		farthest = std::max(farthest, i + nums[i]);//更新最远可到达位置
		if(farthest >= n - 1){
			return true;// 如果能到达最后一个位置
		}
	}

	return true;
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

/*
 * 给定一个整数数组，找到一个具有最大和的子数组，返回其最大和。
 * 每个子数组的数字在数组中的位置应该是连续的。
 * */
int MS::maxSubArray_Dp(vector<int>& nums){
	int n = nums.size();
	vector<int> dp(n + 1);//dp[n]表示第n个数结尾时，最大和
	dp[0] = 0;
	int ans = 0;
	for (int i = 1; i <= n; ++i)
	{
		dp[i] = dp[i - 1] > 0
				? dp[i - 1] + nums[i - 1]
				: nums[i - 1];
		ans = std::max(ans, dp[i]);
	}

	return ans;
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

/*
 * 300. Longest Increasing Subsequence
 * 给定一个整数序列，找到其中最长严格递增子序列（LIS）的长度
 * 思路: 定义 dp[i]表示前i个数中的最长LIS
 * 		转移方程 dp[i] = max(dp[i], dp[j] + 1) 如果nums[j] < nums[i]
 * 		结果：
 * Time:
 * Space:
 * */
int MS::lengthOfLIS(vector<int>& nums){
	if (nums.empty()) return 0;
	int n = nums.size();
	vector<int> dp(n, 1);
	for (int i = 1; i < n; ++i)
	{
		for (int j = 0; j < i; ++j)
		{
			if(nums[j] < nums[i]){
				dp[i] = std::max(dp[i], dp[j] + 1);
			}
		}
	}

	return *max_element(dp.begin(), dp.end());
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

/*
 * 79. Word Search
 * 给定一个 m x n 的字符网格和一个单词 word，判断该单词是否可以通过上下左右相邻的连续字符匹配到。相同单元格内的字母不允许被重复使用。
 * 思路：
 * 		1.回溯
 * 		2.暂时将访问过的board[i][j] = "#"
 * Time:
 * Space:
 * */
bool MS::exist(vector<vector<char>>& board, string word){
	if(word.empty()) return false;
	vector<int> directions = {0, 1, 0, -1, 0};
	int m = board.size();
	int n = board[0].size();
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			if(existDFS(board, word, 0, i, j, directions)){
				return true;
			}
		}
	}

	return false;
}

bool MS::existDFS(vector<vector<char>>& board, string& word, int index, int i, int j, vector<int>& directions){
	int m = board.size();
	int n = board[0].size();
	// 匹配完成
	//if(index == word.size()) return true;
	if(i < 0 || j < 0 || i >= m || j >= n || board[i][j] != word[index]){
		return false;
	}
	if(index == word.size() - 1) return true;
	// 标记已访问（可改成特殊字符）
	char temp = board[i][j];
	board[i][j] = '#';
	bool found = false;
	for (int k = 0; k < 4; ++k)
	{
		int x = i + directions[k];
		int y = j + directions[k + 1];
		found |= existDFS(board, word, index + 1, x, y, directions);
	}

	board[i][j] = temp;

	return found;
}

//200. Number of Islands
int MS::numIslands(vector<vector<char>>& grid){
	vector<int> directions = {0, 1, 0, -1, 0};
	int m = grid.size();
	if(m == 0) return 0;
	int n = grid.front().size();
	int count = 0;
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			if(grid[i][j] == '1'){
				//numIslandsDFS(grid, i, j, directions);
				numIslandsBFS(grid, i, j, directions);
				count++;
			}
		}
	}

	return count;
}

/*
 * 思路：
	遍历网格，当发现一个陆地 '1' 时，启动一个深度优先搜索（DFS）。
	在 DFS 中，将该岛屿的所有陆地（相邻的 '1'）标记为已访问（比如改为 '0'）。
	每次启动 DFS 时，计数器加 1，表示发现了一个新的岛屿
 * */
void MS::numIslandsDFS(vector<vector<char>>& grid, int i, int j, vector<int>& directions){
	int m = grid.size();
	int n = grid.front().size();
	if(i < 0 || i >= m || j < 0 || j >= n || grid[i][j] == '0'){
		return;
	}
	grid[i][j] = '0';
	for (int k = 0; k < 4; ++k)
	{
		int x = i + directions[k];
		int y = j + directions[k + 1];
		numIslandsDFS(grid, x, y, directions);
	}
}

/*
 * BFS: Breath-First-Search
 * 适合图较大，避免递归栈溢出的场景。
 * 思路：
	使用队列模拟广度优先搜索（BFS）逐层访问岛屿。
	每次发现一个新的岛屿，将其所有相邻的陆地 '1' 都标记为 '0'。
 * */
void MS::numIslandsBFS(vector<vector<char>>& grid, int i, int j, vector<int>& directions){
	int m = grid.size();
	int n = grid.front().size();
	queue<pair<int, int>> q;
	q.push({i, j});
	grid[i][j] = '0';
	while (!q.empty()){
		auto [nx, ny] = q.front();
		q.pop();

		for (int k = 0; k < 4; ++k)
		{
			int x = nx + directions[k];
			int y = ny + directions[k + 1];
			if(x < 0 || x >= m || y < 0 || y >= n || grid[x][y] == '0'){
				continue;//🈲️不能直接return，因为还需要接着遍历其他周围点
			}
			q.push({x, y});
			grid[x][y] = '0';
		}
	}
}

//695.Max Area of Island
/*
 *  DFS：
		实现简洁，易于理解。
		适合栈深度较小的场景。
	BFS:
		适合大规模场景：
		避免递归栈溢出，更适合内存受限的场景。

	对整个 grid 进行遍历，每个格子最多访问一次。
	时间复杂度：
		O(R×C)，其中 R 是行数，C 是列数。
	空间复杂度
		DFS: 最差情况下（递归深度为岛屿面积），递归栈的空间复杂度为 O(R×C)。
		BFS: 队列的最大长度为岛屿面积，空间复杂度为 O(R×C)
 * */
int MS::maxAreaOfIsland(vector<vector<int>>& grid){
	int m = grid.size();
	int n = grid.front().size();
	int maxArea = 0;
	//dfs 深度优先搜索
//	for (int i = 0; i < m; ++i)
//	{
//		for (int j = 0; j < n; ++j)
//		{
//			if(grid[i][j] == 1){
//				maxArea = std::max(maxArea, maxAreaOfIslandDFS(grid, i, j));
//			}
//		}
//	}

	//bfs 宽度优先搜索
	const vector<int> directions = {0, 1, 0, -1, 0};
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			if(grid[i][j] == 1){
				maxArea = std::max(maxArea, maxAreaOfIslandBFS(grid, i, j, directions));
			}
		}
	}

	return maxArea;
}

int MS::maxAreaOfIslandDFS(vector<vector<int>>& grid, int i, int j, const vector<int>& directions){
	int m = grid.size();
	int n = grid.front().size();
	if(i < 0 || i >= m || j < 0 || j >= n || grid[i][j] == 0){
		return 0;
	}

	grid[i][j] = 0;
	int count = 1;
	for (int k = 0; k < 4; ++k)
	{
		int x = i + directions[k];
		int y = j + directions[k + 1];
		count += maxAreaOfIslandDFS(grid, x, y, directions);
	}
//	count += maxAreaOfIslandDFS(grid, i - 1, j);
//	count += maxAreaOfIslandDFS(grid, i + 1, j);
//	count += maxAreaOfIslandDFS(grid, i, j + 1);
//	count += maxAreaOfIslandDFS(grid, i, j - 1);

	return count;
}

int MS::maxAreaOfIslandBFS(vector<vector<int>>& grid, int i, int j, const vector<int>& directions){
	int m = grid.size();
	int n = grid.front().size();
	int area = 0;
	queue<pair<int, int>> q;
	q.push({i, j});
	grid[i][j] = 0;
	while(!q.empty()){
		auto [x, y] = q.front();
		q.pop();
		area++;
		for (int k = 0; k < 4; ++k)
		{
			int nx = x + directions[k];
			int ny = y + directions[k + 1];
			if(nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] == 1){
				q.push({nx, ny});
				grid[nx][ny] = 0;
			}
		}
	}

	return area;
}

/*
 * Partition Equal Subset Sum
 *
 * */
bool MS::canPartition(vector<int>& nums){

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

/*
 * 91. Decode Ways
 * 给定一串只包含数字的非空字符串，统计其可能被解码成字母的总数。'A' = 1, 'B' = 2, ... 'Z' = 26。
 * 思路：
 * 		状态：dp[i]表示s[0, i - 1]的解码方法数
 * 		转移方程：dp[i] += dp[i - 1] 当s[i - 1] == '1' ~ '9'
 * 				dp[i] += dp[i - 2] 当s.substring(i - 2, 2) <= 26
 * 		初始化：dp[0] = 1;方便计算  dp[1] = 1;当s[0]不等于 '0'时就有1种
 * Time: O(n)
 * Space: O(n)
 * */
int MS::numDecodings(string s){
	if(s.empty() || s[0] == '0') return 0;
	int n = s.size();
	vector<int> dp(n + 1, 0);
	dp[0] = 1;
	dp[1] = 1;
	for(int i = 2; i <= n; i++){
		//单独解码
		if(s[i - 1] != '0'){
			dp[i] += dp[i - 1];
		}
		// 解码 s[i-2..i-1]
		int twoDigits = stoi(s.substr(i - 2, 2));
		if(twoDigits >= 10 && twoDigits <= 26){
			dp[i] += dp[i - 2];
		}
	}

	return dp[n];
}

/*
 * 115. Distinct Subsequences 【子序列出现的次数】
 * 给定两个字符串S 和 T，问 T 在 S 中出现为子序列的不同方式共有多少种
 * 		这里我们关心的是个数：即有多少种取法会使得取出来的子序列恰好等于T
 *
 * */
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


//1143. Longest Common Subsequence
/*
 * 给定两个字符串 S1和S2，求它们的最长公共子序列的长度
 * 关注：两个串的公共子序列里能达到的最大长度是多少
 * Time: O(m * n)
 * Space: O(m * n)
 * */
int MS::longestCommonSubsequence(string text1, string text2){
	//dp[i][j]表示text1[0...i]中包含text2[0...j]最长公共子序列
	//转移方程：
	// 		如果 text1[i-1] == text2[j-1]，则 dp[i][j] = dp[i-1][j-1] + 1
	//		否则，dp[i][j] = max(dp[i-1][j], dp[i][j-1])
	int n = text1.size();
	int m = text2.size();
	vector<vector<int>> dp(n + 1, vector<int>(m + 1, 0));
	//dp[i][0]和dp[0][j]均等于0
	for (int i = 1; i <= n; ++i)
	{
		for (int j = 1; j <= m; ++j)
		{
			if(text1[i - 1] == text2[j - 1]){
				dp[i][j] = dp[i - 1][j - 1] + 1;
			}
			else{
				dp[i][j] = std::max(dp[i - 1][j], dp[i][j - 1]);
			}
		}
	}

	return dp[n][m];
}


/*
 * 139.Word Break
 * 给定一个字符串 s 和一个包含若干单词的集合 wordDict，判断 s 是否可以被拆分成字典中存在的单词组合
 * 		状态：dp[i]表示s[0, i - 1]是否可以被拆分
 * 		转移方程: dp[i] = dp[j] ,当 s[j, i - 1]在wordDict中存在时
 * 		边界条件: dp[0] = true;
 * Time:O(n^2)
 * Space:O(n)
 * */
bool MS::wordBreak(string s, vector<string>& wordDict){
	if(s.empty() || wordDict.empty()) return false;
	unordered_set<string> set(wordDict.begin(), wordDict.end());
	int n = s.size();
	vector<bool> dp(n + 1, false);//dp[i]表示s[0, i - 1]是否可以被拆分
	dp[0] = true;
	for(int i = 1; i <= n; i++){
		for (int j = 0; j < i; ++j)
		{
			if(dp[j] && set.find(s.substr(j, i - j)) != set.end()){
				dp[i] = dp[j];
				break;
			}
		}
	}

	return dp[n];
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
	/*
		 * 如果键 n 已经存在：
			返回键对应的值的引用（即 freq_map[n]）。
			对引用执行自增操作（++），会增加对应键的值。
			如果键 n 不存在：
			unordered_map 会自动插入一个新的键值对，键为 n，值为默认构造值（对于 int 类型，默认值是 0）。
			然后返回默认值的引用，再执行自增操作。
			这使得 freq_map[n]++ 非常方便地处理频率统计问题，而不需要显式地检查键是否存在
		 * */
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