#include <iostream>
#include <fstream>
#include <string>

#include <Eigen/Dense>
#include <lazycsv.hpp>

#include "util.h"
#include "Gauss_solve.h"

int main(int argc, const char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " input.csv output.csv\n";
        return 1;
    }

    auto ab = load_csv_to_matrix(argv[1]);

    auto solution = Gauss_solve(ab);

    std::ofstream out(argv[2]);
    if (!out.is_open())
    {
        std::cerr << "Cannot open output file: " << argv[2] << "\n";
        return 1;
    }

    GaussMatrix result_matrix(solution.rows(), 1);
    for (int i = 0; i < solution.rows(); ++i)
    {
        result_matrix(i, 0) = solution(i);
    }

    print_matrix_as_csv(out, result_matrix, 10);

    return 0;
}
