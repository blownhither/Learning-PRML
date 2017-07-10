//
//  KDTree.hpp
//  kd-tree
//
//  Created by 马子殷 on 7/9/17.
//  Copyright © 2017 马子殷. All rights reserved.
//

#ifndef KDTree_hpp
#define KDTree_hpp

#include <cstdlib>
#include <string>

struct Node {
    int dim, index;
    double division;        // division coordinate at dim
    Node *l, *r;            // Child nodes
    Node *parent;           // Parent node
};



template<int ndim>
class KDTree {
public:
    KDTree(double **data, int ndata);   // Read data of [ndim, n_data] (row-wise)
    void print();
    
private:
    double **data;
    int ndata;
    int *index;
    Node *head;
    
    void buildTree();
    Node* recBuildTree(int dim, int start, int end);
    int median(int dim, int start, int end);    // modify index
    std::string printPoint(int i);
};

void testKDTree() {
    double a1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    double a2[] = {6, 4, 3, 4, 7, 3, 1, 8, 5};
    double **data = new double* [2];
    data[0] = a1;
    data[1] = a2;
    KDTree<2> kdt(data, 9);
    kdt.print();
}


#endif /* KDTree_hpp */
