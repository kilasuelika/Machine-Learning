#include<iostream>
#include<string>
#include<vector>
#include<gsl/gsl_rng.h>
#include<gsl/gsl_randist.h>
#include<cmath> 
#include<boost/range/numeric.hpp>
#include<random>
#include<iomanip>
#include<numeric>


using namespace std;
using std::string;
using std::vector;

class aux_particle_filter {
private:
	int sizey, n_particles, in;
	vector<double> y, Q, xh;
	vector<vector<double>> W, x;
	const gsl_rng_type *T;
	gsl_rng *rinit, *rp, *rq;
	double *fx, *gx;

public:
	aux_particle_filter(vector<double> yn, long int nt, long int n_particles) {
		//initialize variables.
		this->y=yn;
		this->xh.resize(nt + 1);
		this->n_particles = n_particles;
		this->W.resize(nt+1,vector<double>(this->n_particles,0));
		this->x.resize(nt+1, vector<double>(this->n_particles, 0));
		this->Q.resize(this->n_particles);
		this->sizey = nt + 1;

		//initialize random number generation methods.
		this->T = gsl_rng_default;
		this->rinit = gsl_rng_alloc(this->T);
		this->rp = gsl_rng_alloc(this->T);
		this->rq = gsl_rng_alloc(this->T);
	}

	double f(double x) {
		return 0.97 * x;
	};
	double py(double x, double xn) {
		return gsl_ran_gaussian_pdf(x, 0.69*exp(xn/2));
	};

	vector<double> normalize(vector<double> v) {
		vector<double> v1(v);
		double sum = boost::accumulate(v, 0.0);
		for (auto &i : v1) {
			i /= sum;
		}
		return v1;
	};

	bool run_filter() {
		cout << "Begin..." << endl;
		gsl_rng_env_setup();
		//Initialize x0.
		for (int i = 0; i < this->n_particles; i++) {
			this->x[0][i] = gsl_ran_gaussian(this->rinit, 0.178);
			this->W[0][i] = double(1) / double(this->n_particles);
		};
		const double* p = &Q[0];
		for (int n = 1; n < this->sizey; n++) {
			for (int i = 0; i < n_particles; i++) {
				Q[i] = py(y[n], x[n-1][i])*W[n-1][i];
			};
			gsl_ran_discrete_t *g= gsl_ran_discrete_preproc(this->n_particles, p);
			
			for (int i = 0; i < this->n_particles; i++) {
				//ix is the index variable.
				int ix = gsl_ran_discrete(rq, g);
				double sx = gsl_ran_gaussian(this->rp,1)+ f(x[n - 1][ix]);
				this->x[n][i] = sx;
				W[n][i] = py(y[n],x[n][i])/ py(y[n], f(x[n-1][i]));
			};
		};
		//Normalize Weight matrix.
		for (auto &v : W) {
			v = normalize(v);
		};
		//Generate predicted xhat.
		for (int i = 0; i < sizey;i++) {
			this->xh[i] = inner_product(x[i].begin(),x[i].end(),W[i].begin(),0.0);
		};
		cout << "Auxiliary particle filtering finished. Number of streams: "<< this->n_particles << endl;
		return true;
	};

	vector<double> get_xh() {
		return this->xh;
	};
};


double gy(double x) {
	return exp(x/2)*0.69;
};
int main() {
	vector<double> xn(100), yn(100);
	xn[0] = -0.05;
	default_random_engine dr;
	normal_distribution<> eta(0,0.178);
	normal_distribution<> v(0, 1);
	for (int n = 1; n <= 99; n++) {
		xn[n] = 0.97 * xn[n - 1]+eta(dr);
		yn[n] = 0.69*v(dr)*exp(xn[n] / 2);
	};
	aux_particle_filter pf = aux_particle_filter(yn,99,500);
	pf.run_filter();
	vector<double> xh = pf.get_xh();

	//Print result.
	cout <<setw(10) <<"xn"<<"    "<<"yn"<<"    "<<"xnhat"<<endl;
	for (int n = 1; n <= 99; n++) {
		cout << setw(10)
			<<yn[n] << "    " << xn[n]<<  "    "<< xh[n]<< endl;
	}


	return 0;
}