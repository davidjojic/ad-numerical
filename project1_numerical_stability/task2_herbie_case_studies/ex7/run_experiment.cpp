#include <codi.hpp>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "codipack_fn_original.cpp"
#include "codipack_fn_manual.cpp"
#include "codipack_fn_herbie.cpp"

constexpr std::size_t N = 5;
constexpr std::size_t VARIED_INDEX = 3;
constexpr std::size_t TESTS_PER_SWEEP = 1000;


// ============================================================
// CoDiPack wrappers
// ============================================================

template <typename AD>
void original_wrapper(
    const std::vector<AD>& x,
    std::vector<AD>& y
) {
    codi_func_original(
        x.data(),
        y.data()
    );
}


template <typename AD>
void manual_wrapper(
    const std::vector<AD>& x,
    std::vector<AD>& y
) {
    codi_func_manual(
        x.data(),
        y.data()
    );
}


template <typename AD>
void herbie_wrapper(
    const std::vector<AD>& x,
    std::vector<AD>& y
) {
    codi_func_herbie(
        x.data(),
        y.data()
    );
}


// ============================================================
// Input generation
// ============================================================

std::vector<double> geomspace(
    double start,
    double end,
    std::size_t count
) {
    if (start <= 0.0 || end <= 0.0) {
        throw std::invalid_argument(
            "geomspace requires positive endpoints"
        );
    }

    std::vector<double> values(count);

    if (count == 0) {
        return values;
    }

    if (count == 1) {
        values[0] = start;
        return values;
    }

    const double log_start = std::log(start);
    const double log_end = std::log(end);

    const double step =
        (log_end - log_start)
        / static_cast<double>(count - 1);

    for (std::size_t i = 0; i < count; ++i) {
        values[i] = std::exp(
            log_start
            + static_cast<double>(i) * step
        );
    }

    // Osiguravamo tačne krajnje vrednosti.
    values.front() = start;
    values.back() = end;

    return values;
}


/*
    Generiše:

    -maximum, ..., -minimum,
    +minimum, ..., +maximum

    Tačna nula nije uključena.
*/
std::vector<double> symmetric_geomspace(
    double minimum_absolute_value,
    double maximum_absolute_value,
    std::size_t count
) {
    if (
        minimum_absolute_value <= 0.0
        || maximum_absolute_value <= 0.0
    ) {
        throw std::invalid_argument(
            "Absolute values must be positive"
        );
    }

    if (
        minimum_absolute_value
        > maximum_absolute_value
    ) {
        throw std::invalid_argument(
            "Minimum must not exceed maximum"
        );
    }

    if (count < 2) {
        throw std::invalid_argument(
            "At least two values are required"
        );
    }

    const std::size_t negative_count = count / 2;
    const std::size_t positive_count =
        count - negative_count;

    const std::vector<double> negative_magnitudes =
        geomspace(
            minimum_absolute_value,
            maximum_absolute_value,
            negative_count
        );

    const std::vector<double> positive_values =
        geomspace(
            minimum_absolute_value,
            maximum_absolute_value,
            positive_count
        );

    std::vector<double> values;
    values.reserve(count);

    // Od velike negativne vrednosti ka maloj negativnoj.
    for (
        auto iterator = negative_magnitudes.rbegin();
        iterator != negative_magnitudes.rend();
        ++iterator
    ) {
        values.push_back(-(*iterator));
    }

    // Od male pozitivne ka velikoj pozitivnoj.
    values.insert(
        values.end(),
        positive_values.begin(),
        positive_values.end()
    );

    return values;
}


// ============================================================
// Sweep configuration
// ============================================================

struct SweepConfig {
    std::string name;
    std::vector<double> values;
};


// ============================================================
// Finite differences
// ============================================================

void update_max_difference(
    double first,
    double second,
    std::optional<double>& maximum
) {
    if (
        !std::isfinite(first)
        || !std::isfinite(second)
    ) {
        return;
    }

    const double difference =
        std::abs(first - second);

    if (
        !maximum.has_value()
        || difference > maximum.value()
    ) {
        maximum = difference;
    }
}


std::optional<double> max_vector_difference(
    const std::vector<double>& first,
    const std::vector<double>& second
) {
    std::optional<double> maximum;

    for (std::size_t i = 0; i < N; ++i) {
        update_max_difference(
            first[i],
            second[i],
            maximum
        );
    }

    return maximum;
}


template <typename Jacobian>
std::optional<double> max_jacobian_difference(
    const Jacobian& first,
    const Jacobian& second
) {
    std::optional<double> maximum;

    // i = output
    for (std::size_t i = 0; i < N; ++i) {

        // j = input
        for (std::size_t j = 0; j < N; ++j) {
            update_max_difference(
                first(i, j),
                second(i, j),
                maximum
            );
        }
    }

    return maximum;
}


template <typename Hessian>
std::optional<double> max_hessian_difference(
    const Hessian& first,
    const Hessian& second
) {
    std::optional<double> maximum;

    for (
        std::size_t output = 0;
        output < N;
        ++output
    ) {
        for (
            std::size_t first_input = 0;
            first_input < N;
            ++first_input
        ) {
            for (
                std::size_t second_input = 0;
                second_input < N;
                ++second_input
            ) {
                update_max_difference(
                    first(
                        output,
                        first_input,
                        second_input
                    ),

                    second(
                        output,
                        first_input,
                        second_input
                    ),

                    maximum
                );
            }
        }
    }

    return maximum;
}


// ============================================================
// Maximum difference together with input that produced it
// ============================================================

struct TrackedMaximum {
    std::optional<double> difference;
    double varied_value =
        std::numeric_limits<double>::quiet_NaN();
};


void update_tracked_maximum(
    TrackedMaximum& tracked,
    const std::optional<double>& candidate,
    double varied_value
) {
    if (!candidate.has_value()) {
        return;
    }

    if (
        !tracked.difference.has_value()
        || candidate.value()
            > tracked.difference.value()
    ) {
        tracked.difference = candidate.value();
        tracked.varied_value = varied_value;
    }
}


void print_tracked_maximum(
    const std::string& name,
    const TrackedMaximum& tracked
) {
    std::cout << name << ":\n";

    if (tracked.difference.has_value()) {
        std::cout
            << "  maximum difference = "
            << tracked.difference.value()
            << '\n';

        std::cout
            << "  x[" << VARIED_INDEX << "] = "
            << tracked.varied_value
            << '\n';
    } else {
        std::cout
            << "  no jointly finite values\n";
    }
}


struct SweepStatistics {
    TrackedMaximum primal_manual_original;
    TrackedMaximum primal_herbie_original;
    TrackedMaximum primal_manual_herbie;

    TrackedMaximum first_manual_original;
    TrackedMaximum first_herbie_original;
    TrackedMaximum first_manual_herbie;

    TrackedMaximum second_manual_original;
    TrackedMaximum second_herbie_original;
    TrackedMaximum second_manual_herbie;
};


// ============================================================
// Run one sweep
// ============================================================

void run_sweep(const SweepConfig& sweep) {
    std::filesystem::create_directories("results");

    const std::string output_path =
        "results/" + sweep.name + ".csv";

    std::ofstream csv(output_path);

    if (!csv.is_open()) {
        throw std::runtime_error(
            "Could not open output file: "
            + output_path
        );
    }

    csv << std::setprecision(17);

    csv
        << "test_id,"
        << "varied_index,"
        << "varied_value,"

        << "primal_manual_vs_original,"
        << "primal_herbie_vs_original,"
        << "primal_manual_vs_herbie,"

        << "first_manual_vs_original,"
        << "first_herbie_vs_original,"
        << "first_manual_vs_herbie,"

        << "second_manual_vs_original,"
        << "second_herbie_vs_original,"
        << "second_manual_vs_herbie\n";

    const std::vector<double> base_x = {
        0.5,
        0.5,
        0.5,
        0.5,
        0.5
    };

    using EH = codi::EvaluationHelper;

    SweepStatistics statistics;

    std::size_t saved_tests = 0;
    std::size_t failed_tests = 0;

    const double missing_value =
        std::numeric_limits<double>::quiet_NaN();

    std::cout
        << "\n========================================\n"
        << "RUNNING " << sweep.name << '\n'
        << "========================================\n";

    for (
        std::size_t test_id = 0;
        test_id < sweep.values.size();
        ++test_id
    ) {
        const double varied_value =
            sweep.values[test_id];

        std::vector<double> x = base_x;

        x[VARIED_INDEX] = varied_value;

        try {
            // =================================================
            // Primal
            // =================================================

            std::vector<double> primal_original(N);
            std::vector<double> primal_manual(N);
            std::vector<double> primal_herbie(N);

            codi_func_original(
                x.data(),
                primal_original.data()
            );

            codi_func_manual(
                x.data(),
                primal_manual.data()
            );

            codi_func_herbie(
                x.data(),
                primal_herbie.data()
            );

            // =================================================
            // Jacobians
            // =================================================

            auto jacobian_original =
                EH::createJacobian(N, N);

            auto jacobian_manual =
                EH::createJacobian(N, N);

            auto jacobian_herbie =
                EH::createJacobian(N, N);

            // =================================================
            // Hessians
            // =================================================

            auto hessian_original =
                EH::createHessian(N, N);

            auto hessian_manual =
                EH::createHessian(N, N);

            auto hessian_herbie =
                EH::createHessian(N, N);

            // =================================================
            // CoDiPack differentiation
            // =================================================

            EH::evalJacobianAndHessian(
                original_wrapper<
                    EH::HessianComputationType
                >,
                x,
                N,
                jacobian_original,
                hessian_original
            );

            EH::evalJacobianAndHessian(
                manual_wrapper<
                    EH::HessianComputationType
                >,
                x,
                N,
                jacobian_manual,
                hessian_manual
            );

            EH::evalJacobianAndHessian(
                herbie_wrapper<
                    EH::HessianComputationType
                >,
                x,
                N,
                jacobian_herbie,
                hessian_herbie
            );

            // =================================================
            // Primal differences
            // =================================================

            const auto primal_manual_original =
                max_vector_difference(
                    primal_manual,
                    primal_original
                );

            const auto primal_herbie_original =
                max_vector_difference(
                    primal_herbie,
                    primal_original
                );

            const auto primal_manual_herbie =
                max_vector_difference(
                    primal_manual,
                    primal_herbie
                );

            // =================================================
            // First-order differences
            // =================================================

            const auto first_manual_original =
                max_jacobian_difference(
                    jacobian_manual,
                    jacobian_original
                );

            const auto first_herbie_original =
                max_jacobian_difference(
                    jacobian_herbie,
                    jacobian_original
                );

            const auto first_manual_herbie =
                max_jacobian_difference(
                    jacobian_manual,
                    jacobian_herbie
                );

            // =================================================
            // Second-order differences
            // =================================================

            const auto second_manual_original =
                max_hessian_difference(
                    hessian_manual,
                    hessian_original
                );

            const auto second_herbie_original =
                max_hessian_difference(
                    hessian_herbie,
                    hessian_original
                );

            const auto second_manual_herbie =
                max_hessian_difference(
                    hessian_manual,
                    hessian_herbie
                );

            // =================================================
            // Update global maximums
            // =================================================

            update_tracked_maximum(
                statistics.primal_manual_original,
                primal_manual_original,
                varied_value
            );

            update_tracked_maximum(
                statistics.primal_herbie_original,
                primal_herbie_original,
                varied_value
            );

            update_tracked_maximum(
                statistics.primal_manual_herbie,
                primal_manual_herbie,
                varied_value
            );

            update_tracked_maximum(
                statistics.first_manual_original,
                first_manual_original,
                varied_value
            );

            update_tracked_maximum(
                statistics.first_herbie_original,
                first_herbie_original,
                varied_value
            );

            update_tracked_maximum(
                statistics.first_manual_herbie,
                first_manual_herbie,
                varied_value
            );

            update_tracked_maximum(
                statistics.second_manual_original,
                second_manual_original,
                varied_value
            );

            update_tracked_maximum(
                statistics.second_herbie_original,
                second_herbie_original,
                varied_value
            );

            update_tracked_maximum(
                statistics.second_manual_herbie,
                second_manual_herbie,
                varied_value
            );

            // =================================================
            // Save CSV row
            // =================================================

            csv
                << test_id << ','
                << VARIED_INDEX << ','
                << varied_value << ','

                << primal_manual_original.value_or(
                    missing_value
                ) << ','

                << primal_herbie_original.value_or(
                    missing_value
                ) << ','

                << primal_manual_herbie.value_or(
                    missing_value
                ) << ','

                << first_manual_original.value_or(
                    missing_value
                ) << ','

                << first_herbie_original.value_or(
                    missing_value
                ) << ','

                << first_manual_herbie.value_or(
                    missing_value
                ) << ','

                << second_manual_original.value_or(
                    missing_value
                ) << ','

                << second_herbie_original.value_or(
                    missing_value
                ) << ','

                << second_manual_herbie.value_or(
                    missing_value
                )

                << '\n';

            saved_tests += 1;

        } catch (const std::exception& error) {
            failed_tests += 1;

            std::cerr
                << "Test "
                << test_id
                << " failed for x["
                << VARIED_INDEX
                << "] = "
                << varied_value
                << '\n';

            std::cerr
                << error.what()
                << '\n';
        }

        if (test_id % 100 == 0) {
            std::cout
                << "Completed "
                << test_id
                << "/"
                << sweep.values.size()
                << " tests\n";
        }
    }

    csv.close();

    // ========================================================
    // Terminal summary
    // ========================================================

    std::cout
        << "\n----------------------------------------\n"
        << "FINISHED " << sweep.name << '\n'
        << "----------------------------------------\n";

    std::cout
        << "Total generated tests: "
        << sweep.values.size()
        << '\n';

    std::cout
        << "Saved tests: "
        << saved_tests
        << '\n';

    std::cout
        << "Failed tests: "
        << failed_tests
        << '\n';

    std::cout
        << "\nPRIMAL\n";

    print_tracked_maximum(
        "Manual vs Original",
        statistics.primal_manual_original
    );

    print_tracked_maximum(
        "Herbie vs Original",
        statistics.primal_herbie_original
    );

    print_tracked_maximum(
        "Manual vs Herbie",
        statistics.primal_manual_herbie
    );

    std::cout
        << "\nFIRST ORDER\n";

    print_tracked_maximum(
        "Manual vs Original",
        statistics.first_manual_original
    );

    print_tracked_maximum(
        "Herbie vs Original",
        statistics.first_herbie_original
    );

    print_tracked_maximum(
        "Manual vs Herbie",
        statistics.first_manual_herbie
    );

    std::cout
        << "\nSECOND ORDER\n";

    print_tracked_maximum(
        "Manual vs Original",
        statistics.second_manual_original
    );

    print_tracked_maximum(
        "Herbie vs Original",
        statistics.second_herbie_original
    );

    print_tracked_maximum(
        "Manual vs Herbie",
        statistics.second_manual_herbie
    );

    std::cout
        << "\nResults saved in:\n"
        << output_path
        << '\n';
}


// ============================================================
// Main
// ============================================================

int main() {
    std::cout << std::setprecision(17);

    /*
        small_num_sweep:

        -1e-2 ... -1e-16
        +1e-16 ... +1e-2

        Tačna nula se ne koristi.
    */
    const SweepConfig small_num_sweep = {
        "small_num_sweep",

        symmetric_geomspace(
            1.0e-16,
            1.0e-2,
            TESTS_PER_SWEEP
        )
    };

    /*
        large_num_sweep:

        -1e18 ... -1
        +1 ... +1e18
    */
    const SweepConfig large_num_sweep = {
        "large_num_sweep",

        symmetric_geomspace(
            1.0,
            1.0e18,
            TESTS_PER_SWEEP
        )
    };

    try {
        run_sweep(small_num_sweep);
        run_sweep(large_num_sweep);

    } catch (const std::exception& error) {
        std::cerr
            << "Experiment failed:\n"
            << error.what()
            << '\n';

        return 1;
    }

    std::cout
        << "\n========================================\n"
        << "ALL EXPERIMENTS FINISHED\n"
        << "========================================\n";

    return 0;
}