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
#include <array>
#include <limits>
#include <vector>


template<int ndim>
class KDTree {
public:
    KDTree(double **data, int ndata);   // Read data of [ndim, n_data] (row-wise)
    ~KDTree();
    KDTree(const KDTree& t);            // copy constructor
    void print();
    std::vector<std::vector<double> > knn(const double *data, int k) const;
    
    struct Node {
        int dim, index;
        double division;        // division coordinate at dim
        Node *l, *r;            // Child nodes
        Node *parent;           // Parent node
    };
    
    const double INF = std::numeric_limits<double>::infinity();             // Infinity of double
    const double NEG_INF = -std::numeric_limits<double>::infinity();        // Negative infinity of double
    
private:
    double **data;
    int ndata;
    int *index;
    Node *head;
    Node **inv_ref;
    
    void buildTree();
    Node* recBuildTree(int dim, int start, int end);
//    void recDeleteNode(Node *n);
    int median(int dim, int start, int end);    // modify index
    std::string printPoint(int i);
    Node* find(std::array<double, ndim> &target);
    Node* recFind(std::array<double, ndim> &target, Node *p, std::array<double, ndim> &u_bound, std::array<double, ndim> &l_bound);
    Node* nearestNeighbour(std::array<double, ndim> &target, Node *start);
    double normDistance(int col, const std::array<double, ndim> &target) const;
    bool intersect(std::array<double, ndim>& center, double dist, std::array<double, ndim>& u_bound, std::array<double, ndim>& l_bound, int dim) const;
};


#endif /* KDTree_hpp */
