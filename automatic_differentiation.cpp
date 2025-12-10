#include <iostream>
#include <cmath>

class Var {
public:
    Var(double initial) {
        val = initial;
        gradVal = std::nan("");
    };

    ~Var() = default;

    double getVal() { return val; };
    void setVal(double v) {
        val = v;
    };

    double getGradVal() { return gradVal; };
    void setGradVal(double v) {
        gradVal = v;
    };

    Var add(Var& other) {
        Var z(val + other.val);

        // dz/d_self = 1.0
        children.emplace_back(1.0, &z);

        // dz/d_other = 1.0
        other.children.emplace_back(1.0, &z);

        return z;
    };

    Var subtract(Var& other) {
        Var z(val - other.val);

        // dz/d_self = 1.0
        children.emplace_back(1.0, &z);

        // dz/d_other = -1.0
        other.children.emplace_back(-1.0, &z);

        return z;
    };

    Var multiply(Var& other) {
        Var z(val * other.val);

        // dz/d_self = other.val
        children.emplace_back(other.val, &z);

        // dz/d_other = val
        other.children.emplace_back(val, &z);

        return z;
    };

    Var divide(Var& other) {
        Var z(val / other.val);

        // dz/d_self = 1 / other.val
        children.emplace_back(1.0 / other.val, &z);

        // dz/d_other = -value / other.val^2
        other.children.emplace_back(-val / pow(other.val, 2), &z);

        return z;
    };


    Var exponent(int power) {
        Var z(pow(val, power));

        // dz/d_self = power * val ** (power - 1)
        children.emplace_back(power * pow(val, power - 1), &z);

        return z;
    };

    double grad() {
        if (std::isnan(gradVal)) {
            // Compute derivative with the chain rule
            double sum = 0.0;
            
            for (auto child : children) {
                sum += std::get<0>(child) * std::get<1>(child)->grad();
            }

            gradVal = sum;
        }

        return gradVal;
    };

private:
    double val;
    double gradVal;
    std::vector<std::tuple<double, Var*>> children;
};

int main () {
    Var x(5.0);
    Var y(10.0);

    Var z = x.exponent(2);
    Var f = y.multiply(z);
    f.setGradVal(1.0);

    std::cout << f.getVal() << std::endl;

    std::cout << "df/dx = " << x.grad() << std::endl; // 100
    std::cout << "df/dy = " << y.grad() << std::endl; // 25

    return 0;
}