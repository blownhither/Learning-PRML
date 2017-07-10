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
#include <sstream>

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
    
//    static double* row = this->data[dim];
//    struct {
//        bool operator()(int a, int b) const
//        {
//            return row[a] < row[b];
//        }
//    } comp;
    
    double* row = this->data[dim];
    auto comp = [row](int a, int b) {
        return row[a] < row[b];
    };
    
    std::cout << "sort:";
    for(int i=start; i<end; ++i)
        std::cout << row[index[i]];
    
    std::sort(this->index + start, this->index + end, comp);
    
    std::cout << " then:";
    for(int i=start; i<end; ++i)
        std::cout << row[index[i]];
    std::cout << std::endl;
    
    return (start + end) / 2;
}

template<int ndim>
void KDTree<ndim>::print() {
    std::vector<Node *> vec {this->head};
    while(!vec.empty()) {
        std::vector<Node *> temp;
        std::cout << vec[0]->dim << ": ";
        for(Node *n : vec) {
            std::cout << this->printPoint(this->index[n->index]);
            if (n->l != NULL)
                temp.push_back(n->l);
            if (n->r != NULL)
                temp.push_back(n->r);
        }
        std::cout << std::endl;
        vec = temp;
    }
}

template<int ndim>
std::string KDTree<ndim>::printPoint(int i) {
    std::ostringstream ss;
    ss << "(" << this->data[0][i];
    for(int dim=1; dim<ndim; ++dim) {
        ss << ',' << this->data[dim][i];
    }
    ss << ") ";
    return ss.str();
}
