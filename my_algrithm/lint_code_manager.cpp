//
// Created by xiangxx on 2023/8/17.
//
#include "lint_code_manager.h"

#include <vector>

int LintCodeManager::binarySearch(vector<int> &nums, int target){
	if(nums.empty()){
		return -1;
	}

	int low = 0;
	int high = (int)nums.size() - 1;
	while(low + 1 < high){
		int mid = low + (high - low) / 2;
		if(nums[mid] >= target){
			high = mid;
		}
		else if(nums[mid] < target){
			low = mid;
		}
	}

	if(nums[low] == target){
		return low;
	}

	if(nums[high] == target){
		return high;
	}

	return -1;
}

vector<vector<int>> LintCodeManager::permute(vector<int> &nums){
	vector<vector<int>> result;
	if(nums.empty()){
		result.emplace_back();
		return result;
	}

	vector<int> temp;
	vector<bool> visited(nums.size());
	helper(nums, visited, temp, result);

	return result;
}

void LintCodeManager::helper(vector<int>& nums, vector<bool>& visited, vector<int>& temp, vector<vector<int>>& result)
{
	if(temp.size() == nums.size()){
		//c++里没必要，java里是必须做拷贝的
		//vector<int> v(temp);//拷贝构造新的vector对象
		//vector<int> v = temp;
		//const vector<int>& v(temp);
		//const vector<int>& v = temp;//绑定了引用，并没有创建新的vector对象

		result.push_back(temp);//temp是左值，这里调用了vector<int>的拷贝构造函数
		return;
	}

	for(int i = 0; i < nums.size(); i++){
		if(visited[i]){
			continue;
		}

		temp.push_back(nums[i]);
		visited[i] = true;
		helper(nums, visited, temp, result);
		visited[i] = false;
		temp.pop_back();
	}
}

int LintCodeManager::backPack(int m, vector<int> &a){
	//boolean dp[i][w]表示前i个元素是否能拼成重量w
	int n = a.size();
	//bool dp[n + 1][m + 1];
	std::vector<std::vector<bool>> dp(n + 1, std::vector<bool>(m + 1, false));
	dp[0][0] = true;
	for(int i = 1; i <= m; i++){
		dp[0][i] = false;
	}

	for(int i = 1; i <= n; i++){
		for(int j = 0; j <= m; j++){
			dp[i][j] = dp[i - 1][j];
			if(a[i - 1] <= j){
				//dp[i][j] |= dp[i - 1][j - a[i - 1]];
				dp[i][j] = dp[i][j] | dp[i - 1][j - a[i - 1]];
			}
		}
	}

	while(!dp[n][m]){
		m--;
	}

	return m;
}

int LintCodeManager::backPackII(int m, vector<int> &a, vector<int> &v){
	// write your code here
	if(a.empty()){
		return 0;
	}

	int n = a.size();
	//dp[i][j]表示前i个物品装入大小为j的背包里的最大价值
	int dp[n + 1][m + 1];
	for(int i = 0; i <= m; i++){
		dp[0][i] = 0;
	}

	for(int i = 1; i <= n; i++){
		for(int j = 1; j <= m; j++){
			dp[i][j] = dp[i - 1][j];
			if(a[i - 1] <= j){
				dp[i][j] = std::max(dp[i][j], dp[i - 1][j - a[i - 1]] + v[i - 1]);
			}
		}
	}

	return dp[n][m];
}

int LintCodeManager::kthLargestElement(int k, vector<int> &nums){
	if(nums.empty()){
		return -1;
	}

	return kthHelper(nums, 0, nums.size() - 1, nums.size() - k);
}

int LintCodeManager::kthHelper(vector<int> &nums, int start, int end, int targetIndex){
	int low = start;
	int high = end;
	int temp = nums[low];
	while(low < high){
		while(low < high && nums[high] >= temp){
			high--;
		}
		nums[low] = nums[high];
		while(low < high && nums[low] <= temp){
			low++;
		}
		nums[high] = nums[low];
	}
	nums[low] = temp;
	if(low == targetIndex){
		return nums[low];
	}
	else if(low < targetIndex){
		return kthHelper(nums, low + 1, end, targetIndex);
	}
	else{
		return kthHelper(nums, start, low - 1, targetIndex);
	}
}

vector<vector<int>> LintCodeManager::kSumII(vector<int> &a, int k, int target){
	vector<vector<int>> result;
	if(a.empty()){
		result.emplace_back();
		return result;
	}

	vector<int> temp;
	kSumIIHelper(a, 0, k, target, temp, result);
	return result;
}

void LintCodeManager::kSumIIHelper(vector<int>& a, int offset, int remainCount, int remainTarget, vector<int>& temp,
		vector<vector<int>>& result){
	if(remainCount == 0 && remainTarget == 0){
		result.push_back(temp);
		return;
	}

	if(remainCount <= 0){
		return;
	}

	for(int i = offset; i < a.size(); i++){
		temp.push_back(a[i]);
		kSumIIHelper(a, i + 1, remainCount - 1, remainTarget - a[i], temp, result);
		temp.pop_back();
	}
}

#pragma region sort
void LintCodeManager::mergeSort(vector<int> &a) {
	if(a.empty()){
		return;
	}

	vector<int> temp(a.size());
	mergeSort(a, 0, a.size() - 1, temp);
}

void LintCodeManager::mergeSort(vector<int> &a, int start, int end, vector<int>& temp){
	if(start >= end){
		return;
	}

	int mid = start + (end - start) / 2;
	mergeSort(a, start, mid, temp);
	mergeSort(a, mid + 1, end, temp);
	internalMerge(a, start, end, temp);
}

void LintCodeManager::internalMerge(vector<int> &a, int start, int end, vector<int>& temp){
	int mid = start + (end - start) / 2;
	int i = start;
	int j = mid + 1;
	int k = start;
	while(i <= mid && j <= end){
		temp[k++] = a[i] < a[j] ? a[i++] : a[j++];
	}

	while(i <= mid){
		temp[k++] = a[i++];
	}

	while(j <= end){
		temp[k++] = a[j++];
	}
	/*
	for(int i = start; i <= end; i++){
		a[i] = temp[i];
	}
	*/
	std::copy(temp.begin() + start, temp.begin() + end + 1, a.begin() + start);
}

void LintCodeManager::quickSort(vector<int>& a){
	if(a.empty()){
		return;
	}

	quickSort(a, 0, a.size() - 1);
}

void LintCodeManager::quickSort(vector<int>& a, int start, int end){
	if(start >= end){
		return;
	}

	int pivot = internalQuickSort(a, start, end);
	quickSort(a, start, pivot);
	quickSort(a, pivot + 1, end);
}

int LintCodeManager::internalQuickSort(vector<int>& a, int start, int end){
	int low = start;
	int high = end;
	int temp = a[low];
	while(low < high){
		while(low < high && a[high] >= temp){
			high--;
		}
		a[low] = a[high];
		while(low < high && a[low] <= temp){
			low++;
		}
		a[high] = a[low];
	}

	a[low] = temp;
	return low;
}

void LintCodeManager::heapSort(vector<int>& a){
	if(a.empty()){
		return;
	}
	int length = a.size();
	for(int i = length / 2 - 1; i >= 0; i--){
		heapAdjust(a, i, length);
	}

	for(int i = length - 1; i > 0; i--){
		int temp = a[i];
		a[i] = a[0];
		a[0] = temp;
		heapAdjust(a, 0, i);
	}
}

void LintCodeManager::heapAdjust(vector<int>& nums, int index, int len){
	int largest = index;
	int left = 2 * largest + 1;
	int right = left + 1;
	if(left < len && nums[largest] < nums[left]){
		largest = left;
	}

	if(right < len && nums[largest] < nums[right]){
		largest = right;
	}

	if(largest != index){
		int temp = nums[largest];
		nums[largest] = nums[index];
		nums[index] = temp;
		heapAdjust(nums, largest, len);
	}
}
#pragma endregion
