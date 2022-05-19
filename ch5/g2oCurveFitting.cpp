#include <iostream>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>

using namespace std;


// vertex: 3d vector
class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
	  EIGEN_MAKE_ALIGNED_OPERATOR_NEW	// byte alignment 

	  // override the reset function : to set the original value of the optimized variable 
	  virtual void setToOriginImpl() override {
		    _estimate << 0, 0, 0;
	  }
	
	  // override the plus operator, just plain vector addition : to update
	  virtual void oplusImpl(const double *update) override {
		    _estimate += Eigen::Vector3d(update);
	  }
	
	  // the dummy read/write function - we leave empty
	  virtual bool read(istream &in) {}
	
	  virtual bool write(ostream &out) const {}
};


// edge: 1D error term, connected to exactly one vertex
class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
public:
	  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	
	  CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}
	
	  // define the error term computation
	  virtual void computeError() override {
		    const CurveFittingVertex *v = static_cast<const CurveFittingVertex *> (_vertices[0]);	//vertices[] is an inherited private array that stores all the vertices
		    const Eigen::Vector3d abc = v->estimate();	// estimate() function that returns the _estimate member value
		    _error(0, 0) = _measurement - std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));	
	  }
	
	  // the jacobian
	  virtual void linearizeOplus() override {
		    const CurveFittingVertex *v = static_cast<const CurveFittingVertex *> (_vertices[0]);
		    const Eigen::Vector3d abc = v->estimate();
		    double y = exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
		    _jacobianOplusXi[0] = -_x * _x * y;
		    _jacobianOplusXi[1] = -_x * y;
		    _jacobianOplusXi[2] = -y;
	  }
	
	  virtual bool read(istream &in) {}
	  virtual bool write(ostream &out) const {}

public:
	  double _x;  // x data, note y is given in _measurement
};


int main(int argc, char **argv) {
    double ar = 1.0, br = 2.0, cr = 1.0;         // True value of the parameters
    double ae = 2.0, be = -1.0, ce = 5.0;        // Initial estimate of the parameters
    int N = 100;                                 // number of data points
    double w_sigma = 1.0;                        // noise variance
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng;                                 // OpenCV random number generator

    vector<double> x_data, y_data;      // data points
    for (int i = 0; i < N; i++) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }

    // alias for the long data type name of the solvers
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType;  // block solver : The optimization variable dimension of each error item is 3 (a, b, c) and the error value dimension is 1 (error = y - model)
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // linear solver : Linear solver using dense cholesky decomposition. g2o::LinearSolverDense< MatrixType >. Step 1: create a linear solver, LinearSolver.

    // choose the optimization method from GN, LM, DogLeg
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(
                g2o::make_unique<LinearSolverType>()	// Linear equation solver. Step 1: create a linear solver, LinearSolver.
        )	// Matrix block solver. Step 2: create BlockSolver and initialize it with the linear solver defined above.
    );  // Step 3: create BlockSolver and initialize it with the linear solver defined above.
    // g2o::make_unique -> Like std::make_unique.Constructs an object of type T and wraps it in a std::unique_ptr. std::unique_ptr is a smart pointer that owns and manages another object through a pointer and disposes of that object when the unique_ptr goes out of scope.
    
    
    
    g2o::SparseOptimizer optimizer;     // graph optimizer. Step 4: create the core of graph optimization: sparse optimizer.
    optimizer.setAlgorithm(solver);   	// set the solver algorithm of the optimizer
    optimizer.setVerbose(true);       	// print the results

    // add vertex to the graph. Step 5: define the vertex and edge of the graph and add it to SparseOptimizer.
    CurveFittingVertex *v = new CurveFittingVertex();	// Create one vertex
    v->setEstimate(Eigen::Vector3d(ae, be, ce));		  // The single vertex stores one 3d vector of the estimated model params
    v->setId(0); 				    // Give an id=0 to the single vertex
    optimizer.addVertex(v);	// Add that vertex to our graph

    // add edges to the graph.
    for (int i = 0; i < N; i++) {
        CurveFittingEdge *edge = new CurveFittingEdge(x_data[i]);	// Create multiple edges in a loop
        edge->setId(i);
        edge->setVertex(0, v);                // connect to the vertex object, v component vertex with id=0. Note that there's only one vertex so we use unary edge
        edge->setMeasurement(y_data[i]);      // measurement (Observed value), since edge is associated with error (Observed - Model)
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma)); // set the information matrix: inverse of covariance matrix
        optimizer.addEdge(edge);
    }

    // Perform optimization. Step 6: set the optimization parameters and start the optimization.
    cout << "start optimization" << endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

    // print the results
    Eigen::Vector3d abc_estimate = v->estimate();
    cout << "estimated model: " << abc_estimate.transpose() << endl;

    return 0;
}
