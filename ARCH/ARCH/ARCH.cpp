// ARCH.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
//#define _USE_MATH_DEFINES

#include "pch.h"
#include <iostream>
#include <vector>
#include "nlopt.hpp"
#include <fstream>
#include <cmath>
#include <boost/math/constants/constants.hpp>

using namespace std;
double pi = boost::math::constants::pi<double>();

int numberOfRecords = 0;
double ARCHLL(unsigned n, const double* parm, double *grad, void *data) {
	double LLsum = 0, a_t2, sigma_t2;
	double *x = (double *)data;
	for (int i = 1; i < ::numberOfRecords; i++) {
		a_t2 = pow(*x-parm[0],2);
		sigma_t2 = parm[1]+parm[2]*a_t2;
		//Note here is 0.5, not 1/2.
		LLsum += (-0.5*(log(2*::pi)+log(sigma_t2)+ a_t2 /sigma_t2));
		x++;
	};
	return LLsum;
}

int main()
{
	//Load data.
	vector<double> data;
	ifstream file("data.csv");
	double x;
	while (file>>x) {
		data.emplace_back(x);
	};
	::numberOfRecords = data.size();
	double *adata = &data[0];

	//1. Initialization.
	//Initialize a solver.
	nlopt::opt problem(nlopt::LN_COBYLA,3);
	problem.set_xtol_rel(1e-6);
	//Initialize parameters. ARCH(1): mu, alpha0, alpha1.
	vector<double> parm={0,0.1,0.29};
	double obj=0;

	//2. Settings.
	//Set objective function.
	problem.set_max_objective(ARCHLL, adata);
	//Set lower bounds.
	vector<double> lb{ -1,0,0 };
	vector<double> ub{ 1,1,1 };
	problem.set_lower_bounds(lb);
	problem.set_upper_bounds(ub);

	//3. Run optimization.
	nlopt::result result = problem.optimize(parm, obj);

	//4. Inspect results.
	cout << "Results: " << result << endl;
	cout << "Number of Records: " << ::numberOfRecords << endl;
	cout << "Parameters: ";
	for (int i = 0; i < 3; i++) {
		cout << parm[i] << " ";
	};
	cout << endl;
	cout << "Log likelihood: " << obj << endl;

}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
