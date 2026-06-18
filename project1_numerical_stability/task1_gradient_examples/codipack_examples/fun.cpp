#include <codi.hpp>

#include <cmath>
#include <iostream>
#include <vector>

using Real = codi::RealForward;

Real function(const std::vector<Real>& x) {
    return x[0] * x[0] + x[0] * x[1] + sin(x[2]);
}

int main() {
    std::vector<Real> x = {2.0, 3.0, 0.0};

    std::cout << "f(x) = " << function(x) << "\n\n";

    std::cout << "Gradient:\n";

    for (std::size_t j = 0; j < x.size(); ++j) {
        for (std::size_t i = 0; i < x.size(); ++i) {
            x[i].setGradient(0.0);
        }

        x[j].setGradient(1.0);

        Real y = function(x);

        std::cout << "df/dx[" << j << "] = " << y.getGradient() << '\n';
    }

    return 0;
}