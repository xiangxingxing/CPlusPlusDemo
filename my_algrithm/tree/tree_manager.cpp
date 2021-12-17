//
// Created by DEV on 2021/11/1.
//
#include "tree_node.h"
#include "tree_manager.h"

#include "vector"
#include "stack"
#include "queue"

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

    if (r1 != nullptr && r2 != nullptr) {
        if (r1->val == r2->val) {
            return isSymmetric(r1->left, r2->right) && isSymmetric(r1->right, r2->left);
        }
    }

    return false;
}

bool TreeManager::isValidBST(TreeNode *root) {
    return isValidBST(root, nullptr, nullptr);
}

bool TreeManager::isValidBST(TreeNode *root, int *low, int *high) {
    if (root == nullptr) {
        return true;
    }

    int val = root->val;
    if (low != nullptr && val <= *low) return false;
    if (high != nullptr && val >= *high) return false;


    bool left = isValidBST(root->left, low, &val);
    bool right = isValidBST(root->right, &val, high);

    return left && right;
}


