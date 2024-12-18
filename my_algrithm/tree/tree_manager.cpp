//
// Created by DEV on 2021/11/1.
//
#include "tree_node.h"
#include "tree_manager.h"

#include "vector"
#include "stack"
#include "queue"

#include "string"
using namespace std;

vector<int> TreeManager::preorderTraversal(TreeNode *root) {
    vector<int> v;
    if (root == nullptr) {
        return v;
    }
    TreeNode *p = root;
    stack<TreeNode *> sk;
    while (p != nullptr || !sk.empty()) {
        if (p != nullptr) {
            v.push_back(p->val);
            sk.push(p);
            p = p->left;
        } else {
            p = sk.top();
            sk.pop();
            p = p->right;
        }
    }

    return v;
}

vector<int> TreeManager::inorderTraversal(TreeNode *root) {
    vector<int> v;
    if (root == nullptr) {
        return v;
    }
    TreeNode *p = root;
    stack<TreeNode *> sk;
    while (p != nullptr || !sk.empty()) {
        if (p != nullptr) {
            sk.push(p);
            p = p->left;
        } else {
            p = sk.top();
            sk.pop();
            v.push_back(p->val);
            p = p->right;
        }
    }

    return v;
}

vector<int> TreeManager::postorderTraversal(TreeNode *root) {
    vector<int> v;
    if (root == nullptr) {
        return v;
    }
    TreeNode *p = root;
    stack<TreeNode *> sk;
    TreeNode *visited = nullptr;
    while (p != nullptr || !sk.empty()) {
        if (p != nullptr) {
            sk.push(p);
            p = p->left;
        } else {
            p = sk.top();
            if (p->right && p->right != visited) {
                p = p->right;
                sk.push(p);
                p = p->left;
            } else {
                v.push_back(p->val);
                sk.pop();
                visited = p;
                p = nullptr;
            }
        }
    }

    return v;
}

vector<vector<int>> TreeManager::levelOrder(TreeNode *root) {
    vector<vector<int>> vec;
    if (root == nullptr) {
        return vec;
    }

    TreeNode *p = root;
    queue<TreeNode *> q;
    q.push(p);
    while (!q.empty()) {
        int size = q.size();
        vector<int> v;
        while (size-- > 0) {
            TreeNode *head = q.front();
            q.pop();
            v.push_back(head->val);
            if (head->left != nullptr) {
                q.push(head->left);
            }
            if (head->right != nullptr) {
                q.push(head->right);
            }
        }

        vec.push_back(v);
    }

    return vec;
}

vector<vector<int>> TreeManager::levelOrder2(TreeNode *root) {
	vector<vector<int>> result;
	if(root == nullptr){
		return result;
	}

	TreeNode* p = root;
	std::queue<TreeNode*> queue;
	queue.push(p);
	while(!queue.empty()){
		int size = queue.size();
		vector<int> temp;
		while(size > 0){
			TreeNode* node = queue.front();
			queue.pop();
			if(node->left != nullptr){
				queue.push(node->left);
			}
			if(node->right != nullptr){
				queue.push(node->right);
			}

			temp.push_back(node->val);
			size--;
		}

		result.push_back(temp);
	}

	reverse(result.begin(), result.end());

	return result;
}


TreeNode *TreeManager::deleteNodeInBst(TreeNode *root, int key) {
    if (root == nullptr) {
        return nullptr;
    }

    if (root->val == key) {
        if (root->left == nullptr) {
            return root->right;
        }
        if (root->right == nullptr) {
            return root->left;
        }
        TreeNode *right_min_node = getMin(root->right);
        root->val = right_min_node->val;
        root->right = deleteNodeInBst(root->right, right_min_node->val);
    } else if (root->val < key) {
        root->right = deleteNodeInBst(root->right, key);
    } else {
        root->left = deleteNodeInBst(root->left, key);
    }

    return root;
}

TreeNode *TreeManager::getMin(TreeNode *root) {
    TreeNode *p = root;
    while (p->left != nullptr) {
        p = p->left;
    }

    return p;
}

int TreeManager::maxDepth(TreeNode *root) {
    if (root == nullptr) {
        return 0;
    }

    int left = maxDepth(root->left);
    int right = maxDepth(root->right);

    return max(left, right) + 1;
}

//100
bool TreeManager::isSameTree(TreeNode *p, TreeNode *q) {
    if (p == nullptr && q == nullptr) {
        return true;
    }

    if (p == nullptr || q == nullptr) {
        return false;
    }

    if (p->val != q->val) {
        return false;
    }

    return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
}

// 101
bool TreeManager::isSymmetric(TreeNode *root) {
    return isSymmetric(root, root);
}

bool TreeManager::isSymmetric(TreeNode *r1, TreeNode *r2) {
    if (r1 == nullptr && r2 == nullptr) {
        return true;
    }

    if (r1 == nullptr || r2 == nullptr) {
        return false;
    }

    if(r1->val != r2->val){
        return false;
    }

    return isSymmetric(r1->left, r2->right) && isSymmetric(r1->right, r2->left);
}

bool TreeManager::isValidBST(TreeNode *root) {
    return isValidBST(root, LONG_MIN, LONG_MAX);
}

bool TreeManager::isValidBST(TreeNode *root, long low, long high) {
	if (root == nullptr){
		return true;
	}

	if (root->val <= low || root->val >= high){
		return false;
	}

	return isValidBST(root->left, low, root->val) && isValidBST(root->right, root->val, high);
}

int NOT_BALANCED = -1;

bool TreeManager::isBalanced(TreeNode *root){
	return balancedHelper(root) != NOT_BALANCED;
}

int TreeManager::balancedHelper(TreeNode *root){
	if(root == nullptr) return 0;

	int left = balancedHelper(root->left);
	int right = balancedHelper(root->right);
	if(left == NOT_BALANCED || right == NOT_BALANCED || abs(left - right) > 1){
		return NOT_BALANCED;
	}

	return max(left, right) + 1;
}

// 96
int TreeManager::numTrees(int n){
    //auto dp = new int[n + 1];//dp[n]表示n个数能组成的bst总数
    vector<int> dp(n + 1);
    dp[0] = 1;
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= i; ++j) {
            dp[i] += dp[j - 1] * dp[i - j];
        }
    }

    return dp[n];
}

// 95
vector<TreeNode*> TreeManager::generateTrees(int n) {
    if(n == 0){
        return {};
    }
    return generateTrees(1, n);
}

vector<TreeNode*> TreeManager::generateTrees(int start, int end){
    vector<TreeNode*> ans;
    if(start > end){
        ans.push_back(nullptr);
        return ans;
    }
    for (int i = start; i <= end; ++i) {
        vector<TreeNode*> left_trees = generateTrees(start, i - 1);
        vector<TreeNode*> right_trees = generateTrees(i + 1, end);
        for (const auto &l_item : left_trees){
            for (const auto &r_item : right_trees){
                TreeNode * node = new TreeNode(i, l_item, r_item);
                ans.push_back(node);
            }
        }
    }

    return ans;
}

/*
 * 94 · 二叉树中的最大路径和
 * */
int maxValue = INT_MIN;

int TreeManager::maxPathSum(TreeNode *root){
	maxPathSumHelper(root);
	return maxValue;
}

int TreeManager::maxPathSumHelper(TreeNode* root){
	if(root == nullptr){
		return 0;
	}

	int left = maxPathSumHelper(root->left);
	int right = maxPathSumHelper(root->right);
	int sum = std::max(root->val, std::max(root->val + left, root->val + right));
	maxValue = std::max(maxValue, std::max(sum, root->val + left + right));
	return sum;
}

TreeNode* TreeManager::buildTree(vector<int> &preorder, vector<int> &inorder){
	int index = 0;
	for(auto val : inorder){
		inorderMap[val] = index++;
	}

	return buildTreeHelper(preorder, inorder, 0, inorder.size() - 1);
}

TreeNode* TreeManager::buildTreeHelper(vector<int> &preorder, vector<int> &inorder, int inLeft, int inRight){
	if(inLeft > inRight){
		return nullptr;
	}

	int rootValue = preorder[preIndex];
	auto* root = new TreeNode(rootValue);

	int inIdx = inorderMap[rootValue];
	preIndex++;

	root->left = buildTreeHelper(preorder, inorder, inLeft, inIdx - 1);
	root->right = buildTreeHelper(preorder, inorder, inIdx + 1, inRight);
	return root;
}

//https://www.lintcode.com/problem/480/
//输入：{1,2,3,#,5}
//输出：["1->2->5","1->3"]
vector<string> TreeManager::binaryTreePaths(TreeNode * root) {
    // write your code here
    vector<string> res;
    if (root == nullptr){
        return res;
    }
    if(root->left == nullptr && root->right == nullptr){
        res.push_back(to_string(root->val) + "");
        return res;
    }

    auto left = binaryTreePaths(root->left);
    for (auto & str : left){
        res.push_back(to_string(root->val) + "->" + str);
    }

    auto right = binaryTreePaths(root->right);
    for (auto & str : right){
        res.push_back(to_string(root->val) + "->" + str);
    }

    return res;
}

// Encodes a tree to a single string.
string TreeManager::serialize(TreeNode* root)
{
	if (root == nullptr)
	{
		return "{}";
	}

	vector<TreeNode*> nodes;
	nodes.push_back(root);

	int i = 0;
	while (i < nodes.size())
	{
		if (nodes[i] != nullptr)
		{
			nodes.push_back(nodes[i]->left);
			nodes.push_back(nodes[i]->right);
		}

		i++;
	}

	while (nodes.back() == nullptr)
	{
		nodes.pop_back();
	}

	if (nodes.empty()) {
		return "{}";
	}

//	stringstream ss;
//	ss << "{" << nodes[0]->val;
//	for (int i = 1; i < nodes.size(); i++) {
//		if (nodes[i] == nullptr) {
//			ss << ",#";
//		} else {
//			ss << "," << nodes[i]->val;
//		}
//	}
//	ss << "}";
//
//	return ss.str();

	string res = "{" + to_string(nodes[0]->val);
	for (int j = 1; j < nodes.size(); ++j)
	{
		if (nodes[j] == nullptr)
		{
			res += ",#";
		}else
		{
			res += "," + to_string(nodes[j]->val);
		}
	}

	res += "}";

	return res;
}

// Decodes your encoded data to tree.
TreeNode* TreeManager::deserialize(string data)
{
	if (data == "{}")
	{
		return nullptr;
	}

	vector<string> vals = split(data.substr(1, data.size() - 2), ",");
	TreeNode *root = new TreeNode(atoi(vals[0].c_str()));
	queue<TreeNode *> queue;
	queue.push(root);

	bool is_left_child = true;
	for (int i = 1; i < vals.size(); ++i)
	{
		if (vals[i] != "#")
		{
			auto* n = new TreeNode(atoi(vals[i].c_str()));
			TreeNode* cur_node = queue.front();
			if (is_left_child)
			{
				cur_node->left = n;
			}
			else{
				cur_node->right = n;
			}
			queue.push(n);
		}

		if (!is_left_child)
		{
			queue.pop();
		}

		is_left_child = !is_left_child;
	}

	return root;
}

vector<string> split(const string &str, string delim) {
	vector<string> results;
	int lastIndex = 0, index;
	while ((index = str.find(delim, lastIndex)) != string::npos) {
		results.push_back(str.substr(lastIndex, index - lastIndex));
		lastIndex = index + delim.length();
	}

	if (lastIndex != str.length()) {
		results.push_back(str.substr(lastIndex, str.length() - lastIndex));
	}

	return results;
}