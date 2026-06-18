#include <codi.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using Real = codi::RealForward;

std::vector<Real> function(const std::vector<Real>& x) {
    std::vector<Real> y(2);

    y[0] = x[0] * x[1];
    y[1] = sin(x[0]) + x[2] * x[2];

    return y;
}

int main() {
    std::vector<Real> x = {2.0, 3.0, 0.5};

    const std::size_t numberOfInputs = x.size();
    const std::size_t numberOfOutputs = 2;

    std::vector<std::vector<double>> jacobian(
        numberOfOutputs,
        std::vector<double>(numberOfInputs, 0.0)
    );

    for (std::size_t j = 0; j < numberOfInputs; ++j) {
        for (std::size_t i = 0; i < numberOfInputs; ++i) {
            x[i].setGradient(0.0);
        }

        x[j].setGradient(1.0);

        std::vector<Real> y = function(x);

        for (std::size_t i = 0; i < numberOfOutputs; ++i) {
            jacobian[i][j] = y[i].getGradient();
        }
    }

    std::vector<Real> y = function(x);

    std::cout << "Output:\n";
    for (std::size_t i = 0; i < y.size(); ++i) {
        std::cout << "y[" << i << "] = " << y[i] << '\n';
    }

    std::cout << "\nJacobian:\n";

    for (const auto& row : jacobian) {
        for (double value : row) {
            std::cout << std::setw(12) << value << ' ';
        }
        std::cout << '\n';
    }

    return 0;
}