#include <iostream>
#include <cmath>
#include <iomanip>
using namespace std;
/*
    Approximation of cosine using the Taylor series expansion.
    The series is summed until the absolute value of the next term
    becomes smaller than epsilon.
*/
double approxCos(double x, double epsilon) {
    double result = 1; // first term of the series
    double term = 1;
    int k = 1;
    while (fabs(term) >= epsilon) {
        term *= -x * x / (2*k * (2*k - 1));
        result += term;
        k ++;
    }
    return result;
}
/*
    Approximation of arctangent using the Taylor series expansion.
    The summation continues until the difference between the exact
    value atan(x) and the approximation is less than epsilon.
*/
double approxAtan(double x, double epsilon) {
    double result = 0;
    int sign = 1;
    int i = 1;
    while (fabs(atan(x)-result) >  epsilon) {
        result += sign * pow(x, i) / i;
        sign = -sign;
        i +=2;
    }
    return result;
}
/*
    Approximation of square root using Heron's method.
    Iteration stops when the difference between x and guess^2
    becomes smaller than epsilon.
*/
double approxSqrt(double x, double epsilon) {
    double guess = x;
    double diff = x - guess * guess;

    while (fabs(diff) > epsilon) {
        guess = (guess + x / guess) / 2;
        diff = x - guess * guess;
    }
    return guess;
}

int main() {
    // Error tolerances
    double dv = 2.08* pow(10, -6);
    double dfi = 0.545*pow(10, -6);
    double du = 0.545*pow(10, -6);
    // Table header
    cout << "   fi(x)  |  delta fi   |  ~fi(x) |  ~delta fi   |   u(x)   |   delta u  |  ~u(x)  |   ~delta u    |   v(x)   |   delta v  |   ~v(x)  |  ~delta v    |   z(x)  |  delta z  |  ~z(x) |    ~delta z  " << endl;
    cout << "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;

    for (double x = 0.1; x < 0.21; x += 0.01) {
        // Exact and approximate values of fi(x)
        double exactFi = 2.8*x + sqrt(1 + x);
        double approxFi = 2.8*x + approxSqrt(x+1,dfi);

        // Exact and approximate values of u(x) = cos(fi)
        double exactU = cos(approxFi);
        double approxU = approxCos(approxFi, du);
        
        // Exact and approximate values of v(x) = atan(1.5x + 0.2)
        double exactV = atan(1.5*x+0.2);
        double approxV = approxAtan(1.5*x+0.2, dv);

        cout << setw(7) << fixed << setprecision(3) << exactFi << "   ";
        cout << setw(7) << fixed << setprecision(3) << dfi / pow(10, -6) << "*10^-6" << "  ";
        cout << setw(7) << fixed << setprecision(3) << approxFi << "   ";
        cout << setw(7) << fixed << setprecision(5) << abs(exactFi - approxFi) / pow(10, -8) << "*10^-8" << "  ";
        cout << setw(7) << fixed << setprecision(3) << exactU << "   ";
        cout << setw(7) << fixed << setprecision(3) << du / pow(10, -6) << "*10^-6" << "  ";
        cout << setw(7) << fixed << setprecision(3) << approxU << "   ";
        cout << setw(7) << fixed << setprecision(5) << abs(exactU - approxU) / pow(10, -6) << "*10^-6" << "  ";
        cout << setw(7) << fixed << setprecision(3) << exactV << "   ";
        cout << setw(7) << fixed << setprecision(3) << dv / pow(10, -6) << "*10^-6" << "  ";
        cout << setw(7) << fixed << setprecision(3) << approxV << "   ";
        cout << setw(7) << fixed << setprecision(5) << abs(exactV - approxV) / pow(10, -6) << "*10^-6" << "  ";
        cout << setw(7) << fixed << setprecision(3) << cos(2.8*x + sqrt(1+x))*atan(1.5*x +0.2) << "    ";
        cout << "10^-6      ";
        cout << setw(7) << fixed << setprecision(3) << approxU*approxV << "    ";
        cout << setw(7) << fixed << setprecision(4) << abs(cos(2.8*x + sqrt(1+x))*atan(1.5*x +0.2) - approxU*approxV) / pow(10, -6) << "*10^-6" << endl;
    }

    return 0;
}
