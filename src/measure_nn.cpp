#include "degree_preconditioner.h"
#include "diagonal_preconditioner.h"
#include "example.h"
#include "mf_quantum_graph.h"
#include "neumann_neumann_preconditioner.h"
#include "polynomial_preconditioner.h"
#include <iostream>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>

template<typename Preconditioner>
void measure_mf_example_cg(Example& ex, int N, double eps, int runs) {
	double runtime = 0.0;
	double assemblytime = 0.0;
	int iterations;
	double error;
	for (int i = 0; i < runs; ++i) {
		auto start_assembly = std::chrono::system_clock::now();
		MFQuantumGraph problem(N, ex.vertices, ex.edges);
		auto end_assembly = std::chrono::system_clock::now();
		Eigen::ConjugateGradient<MFQuantumGraph, Eigen::Lower|Eigen::Upper, Preconditioner> solver;
		solver.setTolerance(eps);
		solver.compute(problem);
		auto start_run = std::chrono::system_clock::now();
		Eigen::VectorXd solution = solver.solve(problem.bG);
		auto end_run = std::chrono::system_clock::now();
		assemblytime += std::chrono::duration<double>(end_assembly-start_assembly).count();
		runtime += std::chrono::duration<double>(end_run-start_run).count();
		iterations = solver.iterations();
		error = solver.error();
	}
	std::cout << "assembly time: " << assemblytime/runs << " runtime: " << runtime/runs << " iterations: " << iterations << " error: " << error << "\n";
}

template<typename Preconditioner>
void measure_mf_example_bicgstab(Example& ex, int N, double eps, int runs) {
	double runtime = 0.0;
	double assemblytime = 0.0;
	int iterations;
	double error;
	for (int i = 0; i < runs; ++i) {
		auto start_assembly = std::chrono::system_clock::now();
		MFQuantumGraph problem(N, ex.vertices, ex.edges);
		auto end_assembly = std::chrono::system_clock::now();
		Eigen::BiCGSTAB<MFQuantumGraph, Preconditioner> solver;
		solver.setTolerance(eps);
		solver.compute(problem);
		auto start_run = std::chrono::system_clock::now();
		Eigen::VectorXd solution = solver.solve(problem.bG);
		auto end_run = std::chrono::system_clock::now();
		assemblytime += std::chrono::duration<double>(end_assembly-start_assembly).count();
		runtime += std::chrono::duration<double>(end_run-start_run).count();
		iterations = solver.iterations();
		error = solver.error();
	}
	std::cout << "assembly time: " << assemblytime/runs << " runtime: " << runtime/runs << " iterations: " << iterations << " error: " << error << "\n";
}

int main(int argc, char* argv[]) {

	std::string graph = argv[1];
	int size = std::stoi(argv[2]);
	int logN = std::stoi(argv[3]);
	int runs = 1;
	if (argc == 5)
		runs = std::stoi(argv[4]);

	int N = pow(2, logN)-1+2;
	std::string filename = graph+"_"+argv[2];
	Example ex(filename);
	double eps = sqrt(2.2204e-16);

	std::cout << "CG\n";
	std::cout << "Vanilla\n";
	measure_mf_example_cg<Eigen::IdentityPreconditioner>(ex, N, eps, runs);
	
	std::cout << "\nDegree\n";
	measure_mf_example_cg<DegreePreconditioner<double>>(ex, N, eps, runs);

	std::cout << "\nDiagonal\n";
	measure_mf_example_cg<DiagonalPreconditioner<double>>(ex, N, eps, runs);

	std::cout << "\nPolynomial\n";
	measure_mf_example_cg<PolynomialPreconditioner<double>>(ex, N, eps, runs);

	std::cout << "\nNeumann-Neumann\n";
	measure_mf_example_cg<NeumannNeumannPreconditioner<double>>(ex, N, eps, runs);

	std::cout << "\nBiCGSTAB\n";
	std::cout << "Vanilla\n";
	measure_mf_example_bicgstab<Eigen::IdentityPreconditioner>(ex, N, eps, runs);
	
	std::cout << "\nDegree\n";
	measure_mf_example_bicgstab<DegreePreconditioner<double>>(ex, N, eps, runs);

	std::cout << "\nDiagonal\n";
	measure_mf_example_bicgstab<DiagonalPreconditioner<double>>(ex, N, eps, runs);

	std::cout << "\nPolynomial\n";
	measure_mf_example_bicgstab<PolynomialPreconditioner<double>>(ex, N, eps, runs);

	std::cout << "\nNeumann-Neumann\n";
	measure_mf_example_bicgstab<NeumannNeumannPreconditioner<double>>(ex, N, eps, runs);

	return 0;
}
