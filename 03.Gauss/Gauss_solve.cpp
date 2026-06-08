#include "Gauss_solve.h"
#include <cmath>
#include <stdexcept>

GaussVector Gauss_solve(GaussMatrix &ab)
{
    int n = ab.rows();
    int m = ab.cols();

    if (m != n + 1)
    {
        throw std::runtime_error("Invalid matrix: expected [A|b] format");
    }

    const double EPS = 1e-12;
    for (int col = 0; col < n; ++col)
    {
        int pivot = col;
        for (int i = col + 1; i < n; ++i)
        {
            if (std::abs(ab(i, col)) > std::abs(ab(pivot, col)))
            {
                pivot = i;
            }
        }

        if (std::abs(ab(pivot, col)) < EPS)
        {
            continue;
        }

        if (pivot != col)
        {
            ab.row(pivot).swap(ab.row(col));
        }
        for (int i = col + 1; i < n; ++i)
        {
            double factor = ab(i, col) / ab(col, col);
            for (int j = col; j < m; ++j)
            {
                ab(i, j) -= factor * ab(col, j);
            }
        }
    }
    GaussVector x(n);
    for (int i = n - 1; i >= 0; --i)
    {
        double sum = 0.0;
        for (int j = i + 1; j < n; ++j)
        {
            sum += ab(i, j) * x(j);
        }

        if (std::abs(ab(i, i)) > EPS)
        {
            x(i) = (ab(i, m - 1) - sum) / ab(i, i);
        }
    }

    return x;
}
