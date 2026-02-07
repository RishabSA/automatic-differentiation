#include "Optimizers.hpp"

GradientDescentOptimizer::GradientDescentOptimizer(double lr, NeuralNetwork* model) {
    learning_rate = lr;
    neural_network = model;
};

void GradientDescentOptimizer::optimizeModelWeights() {
    // Backpropagation and Gradient Descent for each parameter
    for (auto& layer : neural_network->layers) {
        Matrix& W = layer.first;
        Matrix& b = layer.second;

        // Update W
        for (int i = 0; i < W.rows; i++) {
            for (int j = 0; j < W.cols; j++) {
                Var& weight_param = W.data[i][j];

                // Partial derivative of the Loss function with respect to the weight parameter
                double gradient = weight_param.getGrad();
                weight_param.setVal(weight_param.getVal() - learning_rate * gradient);
            }
        }

        // Update b
        for (int i = 0; i < b.rows; i++) {
            for (int j = 0; j < b.cols; j++) {
                Var& bias_param = b.data[i][j];

                // Partial derivative of the Loss function with respect to the bias parameter
                double gradient = bias_param.getGrad();
                bias_param.setVal(bias_param.getVal() - learning_rate * gradient);
            }
        }
    }
};

void GradientDescentOptimizer::resetGrad() {
    // Reset gradients and the old graph on everything
    for (auto& layer : neural_network->layers) {
        Matrix& W = layer.first;
        Matrix& b = layer.second;

        W.resetGradAndParents();
        b.resetGradAndParents();
    }
}
