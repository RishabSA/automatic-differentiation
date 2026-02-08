#pragma once

#include <vector>
#include <utility>
#include "NeuralNetwork.hpp"

class GradientDescentOptimizer {
public:
    double learning_rate;
    NeuralNetwork* neural_network;

    GradientDescentOptimizer(double lr, NeuralNetwork* model);

    void optimize();
    
    void resetGrad();
};