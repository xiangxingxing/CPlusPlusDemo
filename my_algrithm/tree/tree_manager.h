//
// Created by DEV on 2021/12/17.
//

#ifndef CPPRACTICE_TREE_MANAGER_H
#define CPPRACTICE_TREE_MANAGER_H

#include <unordered_map>

class TreeManager{
public:
    vector<int> preorderTraversal(TreeNode *root);

    vector<int> inorderTraversal(TreeNode *root);

    vector<int> postorderTraversal(TreeNode *root);

    vector<vector<int>> levelOrder(TreeNode *root);

    vector<vector<int>> levelOrder2(TreeNode *root);

    TreeNode *deleteNodeInBst(TreeNode *root, int key);

    TreeNode *getMin(TreeNode *root);

    int maxDepth(TreeNode *root);

    bool isSameTree(TreeNode *p, TreeNode *q);

    bool isSymmetric(TreeNode *root);

    bool isSymmetric(TreeNode *r1, TreeNode *r2);

    bool isValidBST(TreeNode *root);

    bool isValidBST(TreeNode *root, long low, long high);

	bool isBalanced(TreeNode *root);

	int balancedHelper(TreeNode *root);

    int numTrees(int n);

    vector<TreeNode*> generateTrees(int n);

    vector<TreeNode*> generateTrees(int left, int right);

	int maxPathSum(TreeNode *root);

	int maxPathSumHelper(TreeNode* root);

	/*
	 * 73 · 前序遍历和中序遍历树构造二叉树
	 * */
	unordered_map<int, int> inorderMap;
	int preIndex = 0;
	TreeNode* buildTree(vector<int> &preorder, vector<int> &inorder);
	TreeNode* buildTreeHelper(vector<int> &preorder, vector<int> &inorder, int inLeft, int inRight);

    vector<string> binaryTreePaths(TreeNode * root);

	// Encodes a tree to a single string.
	string serialize(TreeNode* root);

	// Decodes your encoded data to tree.
	TreeNode* deserialize(string data);
};

static vector<string> split(const string &str, string delim);

#endif //CPPRACTICE_TREE_MANAGER_H
