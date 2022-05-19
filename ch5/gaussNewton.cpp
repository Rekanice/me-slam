#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main(int argc, char **argv) {
    double ar = 1.0, br = 2.0, cr = 1.0;    //true value of parameters
    double ae = 2.0, be = -1.0, ce = 5.0;   //initial estimation of parameters
    int N = 100;                            //number of data points
    double w_sigma = 1.0;                   //std deviation of noise
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng;                            //Random number generator from opencv

    //Create the data points
    vector<double> x_data, y_data;  
    for (int i = 0; i < N; i++) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));  // The model, y with noise forms the true data
    }

    // start Gauss−Newton iterations
    int iterations = 100;
    double cost = 0, lastCost = 0;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();  // start timer

    // For each iteration,
    for (int iter = 0; iter < iterations; iter++) {
        Matrix3d H = Matrix3d::Zero(); // Hessian = J^T s^{−1} J in Gauss−Newton
        Vector3d b = Vector3d::Zero(); // bias = -J s^{-1} e
        cost = 0;   // cost(objective) function = sum of errors squared
        
        // Find the sum of squared errorS.
        // For each data point,
        for (int i = 0; i < N; i++) {
            double xi = x_data[i], yi = y_data[i]; // the i−th data
            double error = yi - exp(ae * xi * xi + be * xi + ce);
            Vector3d J; // jacobian
            J[0] = -xi * xi * exp(ae * xi * xi + be * xi + ce);     // de/da  we do the derivative by using the derivative equation (not the process) :P
            J[1] = -xi * exp(ae * xi * xi + be * xi + ce);          // de/db
            J[2] = -exp(ae * xi * xi + be * xi + ce);               // de/dc

            H += inv_sigma * inv_sigma * J * J.transpose();     // Note: inv_sigma is just a scalar & can be placed in front (commutative)
            b += -inv_sigma * inv_sigma * error * J;

            cost += error * error;
        }

        // solve Hx=b
        Vector3d dx = H.ldlt().solve(b);
        if (isnan(dx[0])) {
        cout << "result is nan!" << endl;
        break;
        }
        // If the objective function's current value is greater than its previous value after a step (should be reduced, so this should not happen)
        if (iter > 0 && cost >= lastCost) {
            cout << "cost: " << cost << ">= last cost: " << lastCost << ", break." << endl;
            break;
        }

        // If the objective function is reduced after this iteration (using this set of parameters), update the parameters set
        ae += dx[0];
        be += dx[1];
        ce += dx[2];

        lastCost = cost;

        cout << "total cost: " << cost << ", \t\tupdate: " << dx.transpose() << "\t\testimated params: " << ae << "," << be << "," << ce << endl;
    }

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();  // end timer & display the duration of the Gauss-Newton process
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

    cout << "estimated abc = " << ae << ", " << be << ", " << ce << endl;
    return 0;

}