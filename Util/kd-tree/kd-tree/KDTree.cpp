//
//  KDTree.cpp
//  kd-tree
//
//  Created by 马子殷 on 7/9/17.
//  Copyright © 2017 马子殷. All rights reserved.
//

#include <algorithm>
#include <iostream>
#include <vector>

#include "KDTree.hpp"

template<int ndim>
KDTree<ndim>::KDTree(double **data, int ndata) {
    this->head = NULL;
    this->data = data;
    this->ndata = ndata;
    this->index = new int[ndata];
    for(int i=0; i<ndata; ++i)
        this->index[i] = i;
    
    this->buildTree();
}

template<int ndim>
void KDTree<ndim>::buildTree() {
    this->head = this->recBuildTree(0, 0, this->ndata);
}

template<int ndim>
Node* KDTree<ndim>::recBuildTree(int dim, int start, int end) {
    if(end - start <= 0) {
        return NULL;
    }
    else if(end - start == 1) {
        return new Node{dim, start, this->data[dim][start], NULL, NULL, NULL};
    }
    
    int mid = this->median(dim, start, end);
    int next_dim = (dim + 1) % ndim;
    Node* l = recBuildTree(next_dim, start, mid);
    Node* r = recBuildTree(next_dim, mid + 1, end);
    Node* node  = new Node{dim, mid, this->data[dim][mid], l, r, NULL};
    return node;
}

template<int ndim>
int KDTree<ndim>::median(int dim, int start, int end) {
    //TODO: consider using O(n)
    
    static double* row = this->data[dim];
    struct {
        bool operator()(int a, int b) const
        {
            return row[a] < row[b];
        }
    } comp;
    std::sort(this->index + start, this->index + end, comp);
    return (start + end) / 2;
}

template<int ndim>
void KDTree<ndim>::print() {
    std::vector<Node *> vec {this->head};
    while(!vec.empty()) {
        std::vector<Node *> temp;
        std::cout << vec[0]->dim << ": ";
        for(Node *n : vec) {
            std::cout << n->division << ' ';
            if (n->l != NULL)
                temp.push_back(n->l);
            if (n->r != NULL)
                temp.push_back(n->r);
        }
        std::cout << std::endl;
        vec = temp;
    }
}
