#include "LossFunctions.hpp"

Var MSELoss(Matrix& labels, Matrix& preds) {
    if (labels.rows != preds.rows || labels.cols != preds.cols) {
        throw std::runtime_error("Dimension mismatch when attempting to compute loss");
    }

    Var loss(0.0);
    int N = 0;

    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            Var errors = labels.data[i][j] - preds.data[i][j];
            Var squared_errors = errors.pow(2);
            loss = loss + squared_errors; 
            N++;
        }
    }

    Var total(N);
    loss = loss / total;

    return loss;
};
