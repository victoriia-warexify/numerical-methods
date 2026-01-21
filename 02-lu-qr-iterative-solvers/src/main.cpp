#include <iostream>
#include <fstream>
#include <iomanip>
#include <Eigen>
#include "approach.h"
using namespace std;
using namespace Eigen;

VectorXd exactlySolve(const MatrixXd & A, const VectorXd & b) {
    FullPivLU<MatrixXd> lu(A);
    VectorXd x = lu.solve(b);
    return x;
}

void testFifth(Mt& A, Vc& b, int n, double eps, int rows) {
    Mt A1(rows,rows); Mt A2(rows,rows);
    A.clear();
    b.clear();
    // Fill the diagonal with 1s
    for (int i = 0; i < rows; ++i) {
        A1.getData()[i][i] = 1;
    }
    // Fill the upper triangle with -1s
    for (int i = 0; i < rows; ++i) {
        for (int j = i + 1; j < rows; ++j) {
            A1.getData()[i][j] = -1;
        }
    }
    // Fill the diagonal and lower triangle with 1s
    for (int i = 0; i < rows; ++i) {
        A2.getData()[i][i] = 1;
        for (int j = 0; j < i; ++j) {
            A2.getData()[i][j] = 1;
        }
    }
    // Fill the upper triangle with -1s
    for (int i = 0; i < rows; ++i) {
        for (int j = i + 1; j < rows; ++j) {
            A2.getData()[i][j] = -1;
        }
    }
    A2.multiplyMatrixByScalar(eps*n);
    A = A1.addMatrices(A2);
    Vc bp(rows);
    for (int i = 0; i < rows - 1; i++)
    {
        bp.getVec()[i] = -1;
    }
    bp.getVec()[rows - 1] = 1;
    b = bp;
}

int main() {
    ofstream outputFile("docs/results.txt");
    if (outputFile.is_open()) {
        double epsilon;
        vector<vector<double>> matrixData0 = {{0,2,3}, {1,2,4}, {4,5,6}};
        Mt A0(matrixData0);
        vector<double> vectorData0 = {13,17,32};
        Vc b0(vectorData0);
        Mt L, U, P, Q, R;
        MatrixXd A_0(3,3);
        A_0 << 0,2,3,1,2,4,4,5,6;
        VectorXd b_0(3);
        b_0 << 13,17,32;

        vector<vector<double>> matrixData1 = {{12,1,1}, {1,14,1}, {1,1,16}};
        Mt A1(matrixData1);
        vector<double> vectorData1 = {14,16,18};
        Vc b1(vectorData1);
        MatrixXd A_1(3,3);
        A_1 << 12,1,1,1,14,1,1,1,16;
        VectorXd b_1(3);
        b_1 << 14,16,18;

        vector<vector<double>> matrixData2 = {{-12,1,1}, {1,-14,1}, {1,1,-16}};
        Mt A2(matrixData2);
        vector<double> vectorData2 = {-14,-16,-18};
        Vc b2(vectorData2);
        MatrixXd A_2(3,3);
        A_2 << -12,1,1,1,-14,1,1,1,-16;
        VectorXd b_2(3);
        b_2 << -14,-16,-18;

        vector<vector<double>> matrixData3 = {{-12,13,14}, {15,-14,11}, {14,15,-16}};
        Mt A3(matrixData3);
        vector<double> vectorData3 = {14,16,18};
        Vc b3(vectorData3);
        MatrixXd A_3(3,3);
        A_3 << -12,13,14,15,-14,11,14,15,-16;
        VectorXd b_3(3);
        b_3 << 14,16,18;

        vector<vector<double>> matrixData4 = {{12,11,11}, {11,14,11}, {11,11,16}};
        Mt A4(matrixData4);
        vector<double> vectorData4 = {14,16,18};
        Vc b4(vectorData4);
        MatrixXd A_4(3,3);
        A_4 << 12,11,11,11,14,11,11,11,16;
        VectorXd b_4(3);
        b_4 << 14,16,18;

        outputFile << "Test #0" << endl;
        outputFile << "        Exact solution       |    e   |          Simple Iteration Method              |                 Seidel Method                   | " << endl;
        outputFile << "                             |        |             x               |  error  | iter  |             x                |  error  | iter  | " << endl;
        VectorXd xLU = exactlySolve(A_0, b_0);
        Vc x_exact(xLU.size());
        x_exact = xLU;
        for(int i=2; i<7; ++i) {
            outputFile << setw(7) << fixed << setprecision(7) << x_exact ;
            epsilon = pow(10,-i);
            outputFile << setw(7) << fixed << scientific << setprecision(0) << epsilon << "  ";
            int iter;
            Vc xMPI_approach0 = A0.solveApproachMPI(b0, epsilon, iter);
            outputFile << setw(7) << fixed << setprecision(7) << xMPI_approach0;
            outputFile << setw(7) << fixed << scientific << setprecision(3) << (x_exact - xMPI_approach0).exactNorm();
            outputFile << setw(7) << fixed << iter << "   ";
            Vc xS_approach0 = A0.solveApproachSeidel(b0, epsilon, iter);
            outputFile << setw(7) << fixed << setprecision(7) << xS_approach0;
            outputFile << setw(7) << fixed << scientific << setprecision(3) << (x_exact - xS_approach0).exactNorm() << "   ";
            outputFile << setw(3) << fixed << iter << endl;
        }
        outputFile << "________________________________________________________________________________________________________________________________________" << endl;
        outputFile << "        Exact solution       |             LU Decomposition            |                 QR Decomposition          | " << endl;
        outputFile << "                             |             x               |  error   |             x                |  error  | " << endl;
        outputFile << setw(7) << fixed << setprecision(7) << x_exact ;
        Vc xLU_approach0 = A0.solveApproachLU(b0);
        outputFile << setw(7) << fixed << setprecision(7) << xLU_approach0;
        outputFile << setw(7) << fixed << scientific << setprecision(3) << (x_exact - xLU_approach0).exactNorm() << "  ";
        Vc xQR_approach0 = A0.solveApproachQR(b0);
        outputFile << setw(7) << fixed << setprecision(7) << xQR_approach0;
        outputFile << setw(7) << fixed << scientific << setprecision(3) << (x_exact - xQR_approach0).exactNorm();
        outputFile << endl;outputFile << endl;
        outputFile << "________________________________________________________________________________________________________________________________________" << endl;

        outputFile << "Test #1" << endl;
        outputFile << "        Exact solution       |    e   |          Simple Iteration Method              |                 Seidel Method                   | " << endl;
        outputFile << "                             |        |             x               |  error  | iter  |             x                |  error  | iter  | " << endl;
        xLU = exactlySolve(A_1, b_1);
        x_exact = xLU;
        for(int i=2; i<7; ++i) {
            outputFile << setw(7) << fixed << setprecision(7) << x_exact ;
            epsilon = pow(10,-i);
            outputFile << setw(7) << fixed << scientific << setprecision(0) << epsilon << "  ";
            int iter = 0;
            Vc xMPI_approach1 = A1.solveApproachMPI(b1, epsilon, iter);
            outputFile << setw(7) << fixed << setprecision(7) << xMPI_approach1;
            outputFile << setw(7) << fixed << scientific << setprecision(3) << (x_exact - xMPI_approach1).exactNorm();
            outputFile << setw(7) << fixed << iter << "   ";
            Vc xS_approach1 = A1.solveApproachSeidel(b1, epsilon, iter);
            outputFile << setw(7) << fixed << setprecision(7) << xS_approach1;
            outputFile << setw(7) << fixed << scientific << setprecision(3) << (x_exact - xS_approach1).exactNorm() << "   ";
            outputFile << setw(3) << fixed << iter << endl;
        }
        outputFile << "________________________________________________________________________________________________________________________________________" << endl;
        outputFile << "        Exact solution       |             LU Decomposition            |                 QR Decomposition          | " << endl;
        outputFile << "                             |             x               |  error   |             x                |  error  | " << endl;
        outputFile << setw(7) << fixed << setprecision(7) << x_exact ;
        Vc xLU_approach1 = A1.solveApproachLU(b1);
        outputFile << setw(7) << fixed << setprecision(7) << xLU_approach1;
        outputFile << setw(7) << fixed << scientific << setprecision(3) << (x_exact - xLU_approach1).exactNorm() << "  ";
        Vc xQR_approach1 = A1.solveApproachQR(b1);
        outputFile << setw(7) << fixed << setprecision(7) << xQR_approach1;
        outputFile << setw(7) << fixed << scientific << setprecision(3) << (x_exact - xQR_approach1).exactNorm();
        outputFile << endl; outputFile << endl;
        outputFile << "________________________________________________________________________________________________________________________________________" << endl;

        outputFile << "Test #2" << endl;
        outputFile << "        Exact solution       |    e   |          Simple Iteration Method              |                 Seidel Method                   | " << endl;
        outputFile << "                             |        |             x               |  error  | iter  |             x                |  error  | iter  | " << endl;
        xLU = exactlySolve(A_2, b_2);
        x_exact = xLU;
        for(int i=2; i<7; ++i) {
            outputFile << setw(7) << fixed << setprecision(7) << x_exact ;
            epsilon = pow(10,-i);
            outputFile << setw(7) << fixed << scientific << setprecision(0) << epsilon << "  ";
            int iter = 0;
            Vc xMPI_approach2 = A2.solveApproachMPI(b2, epsilon, iter);
            outputFile << setw(7) << fixed << setprecision(7) << xMPI_approach2;
            outputFile << setw(7) << fixed << scientific << setprecision(3) << (x_exact - xMPI_approach2).exactNorm();
            outputFile << setw(7) << fixed << iter << "   ";
            Vc xS_approach2 = A2.solveApproachSeidel(b2, epsilon, iter);
            outputFile << setw(7) << fixed << setprecision(7) << xS_approach2;
            outputFile << setw(7) << fixed << scientific << setprecision(3) << (x_exact - xS_approach2).exactNorm() << "   ";
            outputFile << setw(3) << fixed << iter << endl;
        }
        outputFile << "________________________________________________________________________________________________________________________________________" << endl;
        outputFile << "        Exact solution       |             LU Decomposition            |                 QR Decomposition          | " << endl;
        outputFile << "                             |             x               |  error   |             x                |  error  | " << endl;
        outputFile << setw(7) << fixed << setprecision(7) << x_exact ;
        Vc xLU_approach2 = A2.solveApproachLU(b2);
        outputFile << setw(7) << fixed << setprecision(7) << xLU_approach2;
        outputFile << setw(7) << fixed << scientific << setprecision(3) << (x_exact - xLU_approach2).exactNorm() << "  ";
        Vc xQR_approach2 = A2.solveApproachQR(b2);
        outputFile << setw(7) << fixed << setprecision(7) << xQR_approach2;
        outputFile << setw(7) << fixed << scientific << setprecision(3) << (x_exact - xQR_approach2).exactNorm();
        outputFile << endl; outputFile << endl;
        outputFile << "________________________________________________________________________________________________________________________________________" << endl;

        outputFile << "Test #3" << endl;
        outputFile << "        Exact solution       |    e   |          Simple Iteration Method              |                 Seidel Method                   | " << endl;
        outputFile << "                             |        |             x               |  error  | iter  |             x                |  error  | iter  | " << endl;
        xLU = exactlySolve(A_3, b_3);
        x_exact = xLU;
        for(int i=2; i<7; ++i) {
            outputFile << setw(7) << fixed << setprecision(7) << x_exact ;
            epsilon = pow(10,-i);
            outputFile << setw(7) << fixed << scientific << setprecision(0) << epsilon << "  ";
            int iter = 0;
            Vc xMPI_approach3 = A3.solveApproachMPI(b3, epsilon, iter);
            outputFile << setw(7) << fixed << setprecision(7) << xMPI_approach3;
            outputFile << setw(7) << fixed << scientific << setprecision(3) << (x_exact - xMPI_approach3).exactNorm();
            outputFile << setw(7) << fixed << iter << "   ";
            Vc xS_approach3 = A3.solveApproachSeidel(b3, epsilon, iter);
            outputFile << setw(7) << fixed << setprecision(7) << xS_approach3;
            outputFile << setw(7) << fixed << scientific << setprecision(3) << (x_exact - xS_approach3).exactNorm() << "   ";
            outputFile << setw(3) << fixed << iter << endl;
        }
        outputFile << "________________________________________________________________________________________________________________________________________" << endl;
        outputFile << "        Exact solution       |             LU Decomposition            |                 QR Decomposition          | " << endl;
        outputFile << "                             |             x               |  error   |             x                |  error  | " << endl;
        outputFile << setw(7) << fixed << setprecision(7) << x_exact ;
        Vc xLU_approach3 = A3.solveApproachLU(b3);
        outputFile << setw(7) << fixed << setprecision(7) << xLU_approach3;
        outputFile << setw(7) << fixed << scientific << setprecision(3) << (x_exact - xLU_approach3).exactNorm() << "  ";
        Vc xQR_approach3 = A3.solveApproachQR(b3);
        outputFile << setw(7) << fixed << setprecision(7) << xQR_approach3;
        outputFile << setw(7) << fixed << scientific << setprecision(3) << (x_exact - xQR_approach3).exactNorm();
        outputFile << endl; outputFile << endl;
        outputFile << "________________________________________________________________________________________________________________________________________" << endl;

        outputFile << "Test #4" << endl;
        outputFile << "        Exact solution        |    e   |          Simple Iteration Method               |                 Seidel Method                    | " << endl;
        outputFile << "                              |        |             x                |  error  | iter  |             x                  |  error  | iter  | " << endl;
        xLU = exactlySolve(A_4, b_4);
        x_exact = xLU;
        for(int i=2; i<7; ++i) {
            outputFile << setw(7) << fixed << setprecision(7) << x_exact ;
            epsilon = pow(10,-i);
            outputFile << setw(7) << fixed << scientific << setprecision(0) << epsilon << "  ";
            int iter = 0;
            Vc xMPI_approach4 = A4.solveApproachMPI(b4, epsilon, iter);
            outputFile << setw(7) << fixed << setprecision(7) << xMPI_approach4;
            outputFile << setw(7) << fixed << scientific << setprecision(3) << (x_exact - xMPI_approach4).exactNorm();
            outputFile << setw(7) << fixed << iter << "   ";
            Vc xS_approach4 = A4.solveApproachSeidel(b4, epsilon, iter);
            outputFile << setw(7) << fixed << setprecision(7) << xS_approach4;
            outputFile << setw(7) << fixed << scientific << setprecision(3) << (x_exact - xS_approach4).exactNorm() << "   ";
            outputFile << setw(3) << fixed << iter << endl;
        }
        outputFile << "________________________________________________________________________________________________________________________________________" << endl;
        outputFile << "        Exact solution        |             LU Decomposition             |                 QR Decomposition          | " << endl;
        outputFile << "                              |             x                |  error   |             x                 |  error  | " << endl;
        outputFile << setw(7) << fixed << setprecision(7) << x_exact ;
        Vc xLU_approach4 = A4.solveApproachLU(b4);
        outputFile << setw(7) << fixed << setprecision(7) << xLU_approach4;
        outputFile << setw(7) << fixed << scientific << setprecision(3) << (x_exact - xLU_approach4).exactNorm() << "  ";
        Vc xQR_approach4 = A4.solveApproachQR(b4);
        outputFile << setw(7) << fixed << setprecision(7) << xQR_approach4;
        outputFile << setw(7) << fixed << scientific << setprecision(3) << (x_exact - xQR_approach4).exactNorm();
        outputFile << endl; outputFile << endl;
        outputFile << "________________________________________________________________________________________________________________________________________" << endl;

        outputFile << "Test #5" << endl;
        for (int rows = 4; rows <= 10; rows++) {
            outputFile << "N: " << rows << endl;
            for (double del = pow(10,-6); del <= pow(10,-3); del *= 10) {
                outputFile << setw(7) << fixed << scientific << setprecision(0) << "matrix eps: " << del << endl;
                Mt A5; Vc b5;
                testFifth(A5, b5, 10, del, rows);
                VectorXd b_5 = Map<VectorXd>(b5.getdata(), b5.getSize());

                // Explicitly assign A_5 from A5
                MatrixXd A_5(A5.numCols(), A5.numRows());
                for (int i = 0; i < A5.numCols(); ++i) {
                    for (int j = 0; j < A5.numRows(); ++j) {
                        A_5(i, j) = A5.getData()[j][i];
                    }
                }
                A_5.transposeInPlace();
                VectorXd xLU = exactlySolve(A_5, b_5);
                Vc x_exact(xLU.size());
                x_exact = xLU;
                outputFile << setw(7) << fixed << setprecision(7) << "Exact solution: " << x_exact << endl;
                for(int i=2; i<7; ++i) {
                    epsilon = pow(10,-i);
                    outputFile << "  " << setw(7) << fixed << scientific << setprecision(0) << "epsilon: " << epsilon << " ";
                    int iter = 0;
                    Vc xMPI_approach5 = A5.solveApproachMPI(b5, epsilon, iter);
                    outputFile << setw(7) << fixed << setprecision(7) << "MPI: " << xMPI_approach5 << " ";
                    outputFile << setw(7) << fixed << scientific << setprecision(3) << "Error: " << (x_exact - xMPI_approach5).exactNorm() << " ";
                    outputFile << setw(7) << fixed << "Iterations: " << iter << endl;
                    outputFile << "                   ";
                    Vc xS_approach5 = A5.solveApproachSeidel(b5, epsilon, iter);
                    outputFile << setw(7) << fixed << setprecision(7) << "Seidel: " << xS_approach5 << " ";
                    outputFile << setw(7) << fixed << scientific << setprecision(3) << "Error: " << (x_exact - xS_approach5).exactNorm() << " ";
                    outputFile << setw(7) << fixed << "Iterations: " << iter << endl;
                }
                Vc xLU_approach5 = A5.solveApproachLU(b5);
                outputFile << setw(7) << fixed << setprecision(7) << "LU: " << xLU_approach5 << endl;
                outputFile << setw(7) << fixed << scientific << setprecision(3) << "Error: " << (x_exact - xLU_approach5).exactNorm() << endl;

                Vc xQR_approach5 = A5.solveApproachQR(b5);
                outputFile << setw(7) << fixed << setprecision(7) << "QR: " << xQR_approach5 << endl;
                outputFile << setw(7) << fixed << scientific << setprecision(3) << "Error: " << (x_exact - xQR_approach5).exactNorm()<< endl;

            }
            outputFile << endl;
        }
        outputFile.flush(); // Force flushing buffered data to the file
        outputFile.close();
    }
    else {
        cout << "Unable to open file for writing!" << endl;
        return 1;
    }
    return 0;
}
