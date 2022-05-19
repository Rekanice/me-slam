#include <iostream>
using namespace std;
#include <ctime>
// Eigen core
#include "Eigen/Core"
// Algebraic operations of dense matrices (inverse, eigenvalues, etc.)
#include "Eigen/Dense"
using namespace Eigen;

#define MATRIX_SIZE 50

/*
This program demonstrates the use of the basic Eigen type
*/

int main(int argc, char **argv) {
    // All vectors and matrices in Eigen are Eigen::Matrix, which is a template
    // class. Its first three parameters are: data type, row, column Declare a 2∗3
    // float matrix
    Matrix<float, 2, 3> matrix_23;

    // At the same time, Eigen provides many built−in types via typedef, but the
    // bottom layer is still Eigen::Matrix. For example, Vector3d is essentially
    // Eigen::Matrix<double, 3, 1>, which is a three−dimensional vector.
    Vector3d v_3d;
    // This is the same
    Matrix<float, 3, 1> vd_3d;

    // Matrix3d is essentially Eigen::Matrix<double, 3, 3>
    Matrix3d matrix_33 = Matrix3d::Zero(); // initialized to zero
    // If you are not sure about the size of the matrix, you can use a matrix of 
    // dynamic size
    Matrix<double, Dynamic, Dynamic> matrix_dynamic;
    // simpler
    MatrixXd matrix_x;
    // There are still many types of this kind. We don't list them one by one.

    // Here is the operation of the Eigen matrix
    // input data (initialization)
    matrix_23 << 1, 2, 3, 4, 5, 6;
    // output
    cout << "matrix 2x3 from 1 to 6: \n" << matrix_23 << "\n\n" << endl;

    // Use () to access elements in the matrix
    cout << "print matrix 2x3: " << endl;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++)
            cout << matrix_23(i, j) << "\t";
        cout << endl;
    }

    // We can easily multiply a matrix with a vector (but actually still matrices and matrices)
    v_3d << 3, 2, 1;
    vd_3d << 4, 5, 6;

    // In Eigen you can't mix two different types of matrices, like this is
    // wrong Matrix<double, 2, 1> result_wrong_type = matrix_23 * v_3d;
    // It should be explicitly converted
    Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
    cout << "[1,2,3;4,5,6]*[3,2,1]=" << result.transpose() << "\n" << endl;

    Matrix<float, 2, 1> result2 = matrix_23 * vd_3d;
    cout << "[1,2,3;4,5,6]*[4,5,6]: " << result2.transpose() << "\n" << endl;

    // Also you can't misjudge the dimensions of the matrix
    // Try canceling the comments below to see what Eigen will report.
    // Eigen::Matrix<double, 2, 3> result_wrong_dimension =
    // matrix_23.cast<double>() ∗ v_3d;
    
    // some matrix operations
    // The basic operations are not demonstrated, just use +−∗/ operators.
    matrix_33 = Matrix3d::Random(); // Random Number Matrix
    cout << "random matrix: \n" << matrix_33 << "\n" << endl;
    cout << "transpose: \n" << matrix_33.transpose() << "\n" << endl;
    cout << "sum: " << matrix_33.sum() << "\n" << endl;
    cout << "trace: " << matrix_33.trace() << "\n" << endl;
    cout << "times 10: \n" << 10 * matrix_33 << "\n" << endl;
    cout << "inverse: \n" << matrix_33.inverse() << "\n" << endl;
    cout << "det: " << matrix_33.determinant() << "\n\n" << endl;
    
    // Eigenvalues
    // Real symmetric matrix can guarantee successful diagonalization
    SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);
    cout << "Eigen values = \n" << eigen_solver.eigenvalues() << "\n" << endl;
    cout << "Eigen vectors = \n" << eigen_solver.eigenvectors() << "\n\n" << endl;
    
    // Solving equations
    // We solve the equation of matrix_NN ∗ x = v_Nd
    // The size of N is defined in the previous macro, which is generated by a
    // random number Direct inversion is the most direct, but the amount of
    // inverse operations is large.
    
    Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN = MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    matrix_NN = matrix_NN * matrix_NN.transpose(); // Guarantee semi−positive definite
    Matrix<double, MATRIX_SIZE, 1> v_Nd = MatrixXd::Random(MATRIX_SIZE, 1);
    
    clock_t time_stt = clock(); // timing
    // Direct inversion
    Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
    cout << "time of normal inverse is "
    << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;
    cout << "x = " << x.transpose() << "\n" << endl;
    
    // Usually solved by matrix decomposition, such as QR decomposition, the speed
    // will be much faster
    time_stt = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    cout << "time of Qr decomposition is "
    << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;
    cout << "x = " << x.transpose() << "\n\n" << endl;
    
    // For positive definite matrices, you can also use cholesky decomposition to
    // solve equations.
    time_stt = clock();
    x = matrix_NN.ldlt().solve(v_Nd);
    cout << "time of ldlt decomposition is "
    << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;
    cout << "x = " << x.transpose() << "\n\n" << endl;
    
    return 0;
}