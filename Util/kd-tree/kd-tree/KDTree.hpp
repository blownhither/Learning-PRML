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
};

void testKDTree() {
    double p[] = {8, 7, 6, 5, 4, 3, 2, 1};
    double **data = new double*[2];
    data[0] = data[1] = p;
    KDTree<2> kdt(data, 8);
    kdt.print();
}


#endif /* KDTree_hpp */
