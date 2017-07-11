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
    // Build KDTree with given col-wise data, length specified as ndata
    
    this->head = NULL;
    this->inv_ref = NULL;
    this->data = data;
    this->ndata = ndata;
    this->index = new int[ndata];
    for(int i=0; i<ndata; ++i)
        this->index[i] = i;
    this->buildTree();
}

template<int ndim>
void KDTree<ndim>::buildTree() {
    // Call rec procedure to build tree
    this->inv_ref = new Node*[this->ndata];
    this->head = this->recBuildTree(0, 0, this->ndata);
}

template<int ndim>
Node* KDTree<ndim>::recBuildTree(int dim, int start, int end) {
    // Build tree between index[start : end)
    if(end - start <= 0) {
        return NULL;
    }
    else if(end - start == 1) {
        Node* n = new Node{dim, start, this->data[dim][start], NULL, NULL, NULL};
        this->inv_ref[start] = n;
        return n;
    }
    
    int mid = this->median(dim, start, end);
    int next_dim = (dim + 1) % ndim;
    Node* l = recBuildTree(next_dim, start, mid);
    Node* r = recBuildTree(next_dim, mid + 1, end);
    Node* node  = new Node{dim, mid, this->data[dim][mid], l, r, NULL};
    if(l != NULL)
        l->parent = node;
    if(r != NULL)
        r->parent = node;
    this->inv_ref[mid] = node;
    return node;
}

template<int ndim>
int KDTree<ndim>::median(int dim, int start, int end) {
    // Rearrange this->index, so that data[:, index[start : end)] is in order, the median index is returned
    //TODO: consider using O(n)
    double* row = this->data[dim];
    auto comp = [row](int a, int b) {
        return row[a] < row[b];
    };
    std::sort(this->index + start, this->index + end, comp);
    return (start + end) / 2;
}

template<int ndim>
void KDTree<ndim>::print() {
    // Print the tree structure
    std::vector<Node *> vec {this->head};
    while(!vec.empty()) {
        std::vector<Node *> temp;
        std::cout << vec[0]->dim << ": ";
        for(Node *n : vec) {
            std::cout << this->printPoint(this->index[n->index]) << " ";
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
    // Return string of a point with format "(dim0, dim1, ...)"
    std::ostringstream ss;
    ss << "(" << this->data[0][i];
    for(int dim=1; dim<ndim; ++dim) {
        ss << ',' << this->data[dim][i];
    }
    ss << ")";
    return ss.str();
}

template<int ndim>
std::vector<std::vector<double> > KDTree<ndim>::knn(const double *data, int k) const {
    
}


