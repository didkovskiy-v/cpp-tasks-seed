#include <gtest/gtest.h>
#include <random>
#include <Eigen/Dense>
#include <cmath>

#include "Gauss_solve.h"
#include "util.h"

void CheckResidual(const GaussMatrix& A, const GaussVector& x, const GaussVector& b, double eps = 1e-6)
{
    GaussVector residual = A * x - b;
    EXPECT_LT(residual.norm(), eps);
}

TEST(GaussSolve, SmallSystem_2x2_Handcrafted)
{
    GaussMatrix ab(2, 3);
    ab << 2, 1, 5,
    1, -1, 1;

    GaussMatrix ab_copy = ab;
    auto sol = Gauss_solve(ab_copy);

    EXPECT_NEAR(sol(0), 2.0, 1e-8);
    EXPECT_NEAR(sol(1), 1.0, 1e-8);

    GaussMatrix A = ab.leftCols(2);
    GaussVector b = ab.rightCols(1);
    CheckResidual(A, sol, b);
}

TEST(GaussSolve, IdentityMatrix_NxN)
{
    const int n = 10;
    GaussMatrix I = GaussMatrix::Identity(n, n);
    GaussVector b = GaussVector::LinSpaced(n, 1.0, 10.0);

    GaussMatrix ab(n, n + 1);
    ab << I, b;

    GaussMatrix ab_copy = ab;
    auto sol = Gauss_solve(ab_copy);

    for (int i = 0; i < n; ++i)
    {
        EXPECT_NEAR(sol(i), b(i), 1e-8);
    }
}

TEST(GaussSolve, LargeRandomSystem_100x100)
{
    const int n = 100;
    std::mt19937 gen(12345);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    GaussMatrix A(n, n);
    GaussVector x_true(n);

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            A(i, j) = dist(gen);
        }
        x_true(i) = dist(gen);
    }

    for (int i = 0; i < n; ++i)
    {
        A(i, i) += 10.0 * n;
    }

    GaussVector b = A * x_true;

    GaussMatrix ab(n, n + 1);
    ab << A, b;

    GaussMatrix ab_copy = ab;
    auto sol = Gauss_solve(ab_copy);

    CheckResidual(A, sol, b, 1e-4);

    EXPECT_LT((sol - x_true).norm() / x_true.norm(), 1e-4);
}

TEST(GaussSolve, IllConditionedSystem)
{
    GaussMatrix ab(2, 3);
    ab << 1, 1,           2,
    1, 1.0000001,   2.0000001;

    GaussMatrix ab_copy = ab;
    auto sol = Gauss_solve(ab_copy);

    EXPECT_NEAR(sol(0), 1.0, 1e-3);
    EXPECT_NEAR(sol(1), 1.0, 1e-3);
}

TEST(GaussSolve, SingleEquation_1x1)
{
    GaussMatrix ab(1, 2);
    ab << 5, 10;

    GaussMatrix ab_copy = ab;
    auto sol = Gauss_solve(ab_copy);

    EXPECT_NEAR(sol(0), 2.0, 1e-8);
}

TEST(GaussSolve, ZeroPivotHandling)
{
    GaussMatrix ab(3, 4);
    ab << 0, 2, 3, 8,
    0, 0, 5, 10,
    0, 0, 0, 0;

    GaussMatrix ab_copy = ab;
    EXPECT_NO_THROW(Gauss_solve(ab_copy));
}
