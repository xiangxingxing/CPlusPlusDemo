//
// Created by DEV on 2021/12/17.
//

#ifndef CPPRACTICE_TREE_MANAGER_H
#define CPPRACTICE_TREE_MANAGER_H

class TreeManager{
public:
    vector<int> preorderTraversal(TreeNode *root);

    vector<int> inorderTraversal(TreeNode *root);

    vector<int> postorderTraversal(TreeNode *root);

    vector<vector<int>> levelOrder(TreeNode *root);

    TreeNode *deleteNodeInBst(TreeNode *root, int key);

    TreeNode *getMin(TreeNode *root);

    int maxDepth(TreeNode *root);

    bool isSameTree(TreeNode *p, TreeNode *q);

    bool isSymmetric(TreeNode *root);

    bool isSymmetric(TreeNode *r1, TreeNode *r2);

    bool isValidBST(TreeNode *root);

    bool isValidBST(TreeNode *root, int *low, int *high);

    int numTrees(int n);

    vector<TreeNode*> generateTrees(int n);

    vector<TreeNode*> generateTrees(int left, int right);

    vector<string> binaryTreePaths(TreeNode * root);
};

#endif //CPPRACTICE_TREE_MANAGER_H
