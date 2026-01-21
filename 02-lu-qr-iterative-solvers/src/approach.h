#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen>

using namespace std;

// Function for calculating square root using Heron's method
double approxSqrt(double x) {
    double guess = x;
    const double epsilon = 1e-10;
    double diff = x - guess * guess;

    while (fabs(diff) > epsilon) {
        guess = (guess + x / guess) / 2;
        diff = x - guess * guess;
    }
    return guess;
}

class Vc {
private:
    vector<double> vec;
public:
    // Default constructor
    Vc() : vec(vector<double>()) {}

    // Constructor with given coordinates
    Vc(const vector<double>& coords) : vec(coords) {}

    // Constructor for initializing a vector of a specified length to 0.0
    Vc(int size) : vec(vector<double>(size, 0.0)) {}

    // Method for getting a pointer to vector data
    double* getdata() {
        return vec.data();
    }

    // Clear vector
    void clear() {
        vec.clear();
    }

    // Method for obtaining the dimension of a vector
    int getSize() const {
        return vec.size();
    }

    // Get vector coordinate by index
    double & getCoords(int index) {
        return vec[index];
    }

    // Method for obtaining vector data
    vector<double>& getVec() {
        return vec;
    }

    // Function for calculating infinity norm of a vector
    double vecInfNorm() const {
        double maxElement = 0.0;
        for (double val : vec) {
            double absVal = abs(val);
            if (absVal > maxElement) {
                maxElement = absVal;
            }
        }
        return maxElement;
    }

    // Overloading the addition operator for vectors
    Vc operator+(const Vc& other) const {
        if (getSize() == other.getSize()) {
            vector<double> result;
            for (int i = 0; i < getSize(); ++i) {
                result.push_back(vec[i] + other.vec[i]);
            }
            return Vc(result);
        } else return Vc(vector<double>());
    }

    // Overloading the subtraction operator for vectors
    Vc operator-(const Vc& other) const {
        if (getSize() == other.getSize()) {
            vector<double> result;
            for (int i = 0; i < getSize(); ++i) {
                result.push_back(vec[i] - other.vec[i]);
            }
            return Vc(result);
        } else return Vc(vector<double>());
    }

    // Overloading scalar multiplication operator
    Vc operator*(double scalar) const {
        vector<double> result;
        for (double coord : vec) {
            result.push_back(coord * scalar);
        }
        return Vc(result);
    }

    // Overloading output operator
    friend ostream& operator<<(ostream& os, Vc& vec) {
        for (int i = 0; i < vec.getSize(); ++i) {
            os << vec.getCoords(i) << " ";
        }
        return os;
    }

    // Overloading assignment operator from Eigen vector
    Vc& operator=(const Eigen::VectorXd& other) {
        for (int i = 0; i < other.size(); ++i) {
            getCoords(i) = other(i);
        }
        return *this;
    }

    // Method for calculating the norm of a vector
    double norm() const {
        double norm = 0.0;
        for (double val : vec) {
            norm += val * val;
        }
        norm = approxSqrt(norm);
        return norm;
    }

    // Normalize vector
    void normalize() {
        double normVal = norm();
        for (double& val : vec) {
            val /= normVal;
        }
    }

    // Exact Euclidean norm
    double exactNorm() const {
        double norm = 0.0;
        for (double val : vec) {
            norm += val * val;
        }
        norm = sqrt(norm);
        return norm;
    }
};

class Mt {
private:
    vector<vector<double>> matrix;
public:
    // Default constructor
    Mt() {}

    // Constructor from given matrix
    Mt(vector<vector<double>>& input_matrix) : matrix(input_matrix) {}

    // Constructor initializing matrix with zeros
    Mt(int rows, int cols) : matrix(vector<vector<double>>(rows, vector<double>(cols, 0.0))) {}

    // Method for getting a pointer to matrix data
    double* data() {
        return matrix[0].data();
    }

    // Clear matrix
    void clear() {
        matrix.clear();
    }

    // Get matrix data
    vector<vector<double>>& getData() {
        return matrix;
    }

    // Get number of rows
    int numRows() const {
        return matrix.size();
    }

    // Get number of columns
    int numCols() const {
        if (matrix.size() > 0) {
            return matrix[0].size();
        } else {
            return 0;
        }
    }

    // Resize matrix
    void resize(int rows, int cols) {
        matrix.resize(rows, vector<double>(cols, 0.0));
    }

    // Get size of a specific row
    int getRowSize(int rowNumber) const {
        if (rowNumber >= 0 && rowNumber < matrix.size()) {
            return matrix[rowNumber].size();
        } else {
            return -1;
        }
    }

    // Compute infinity norm of matrix
    double matrixInfNorm() const {
        double rows = matrix.size();
        double cols = matrix[0].size();
        double maxRowSum = 0.0;

        for (int i = 0; i < rows; ++i) {
            double rowSum = 0.0;
            for (int j = 0; j < cols; ++j) {
                rowSum += abs(matrix[i][j]);
            }
            if (rowSum > maxRowSum) {
                maxRowSum = rowSum;
            }
        }

        return maxRowSum;
    }

    // Multiply two matrices
    static Mt multiplyMatrices(Mt A, Mt& B) {
        vector<vector<double>>& AData = A.getData();
        vector<vector<double>>& BData = B.getData();

        int rowsA = AData.size();
        int colsA = AData[0].size();
        int rowsB = BData.size();
        int colsB = BData[0].size();

        vector<vector<double>> result(rowsA, vector<double>(colsB, 0));

        for (int i = 0; i < rowsA; ++i) {
            for (int j = 0; j < colsB; ++j) {
                for (int k = 0; k < colsA; ++k) {
                    result[i][j] += AData[i][k] * BData[k][j];
                }
            }
        }
        return Mt(result);
    }

    // Swap two rows of the matrix
    void swapRows(int row1, int row2) {
        vector<double> temp = matrix[row1];
        matrix[row1] = matrix[row2];
        matrix[row2] = temp;
    }

    // Multiply matrix by vector
    Vc multiplyMtVc(Vc& vec)  {
        int Rows = numRows();
        int Cols = numCols();
        vector<double> result(Rows, 0.0);
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                result[i] += matrix[i][j] * vec.getCoords(j);
            }
        }

        return result;
    }

    // Transpose matrix
    Mt transposeMt()  {
        int rows = matrix.size();
        int cols = matrix[0].size();

        Mt result(cols, rows);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result.getData()[j][i] = matrix[i][j];
            }
        }

        return result;
    }

    // Check if matrix is diagonally dominant
    bool isDiagonalDominant() {
        int rows = matrix.size();
        for (int i = 0; i < rows; ++i) {
            int diagonalElement = abs(matrix[i][i]);
            int sum = 0;
            for (int j = 0; j < rows; ++j) {
                if (j != i) {
                    sum += abs(matrix[i][j]);
                }
            }
            if (diagonalElement <= sum) {
                return false;
            }
        }
        return true;
    }

    // Add matrices
    Mt addMatrices(Mt& matrix1) {
        if (numRows() != matrix1.numRows() || numCols() != matrix1.numCols()) {
            return Mt();
        }

        Mt result(numRows(), numCols());

        for (int i = 0; i < numRows(); ++i) {
            for (int j = 0; j < numCols(); ++j) {
                result.getData()[i][j] = matrix[i][j] + matrix1.getData()[i][j];
            }
        }
        return result;
    }

    // Multiply matrix by scalar
    void multiplyMatrixByScalar (double scalar) {
        for (int i = 0; i < numRows(); ++i) {
            for (int j = 0; j < numCols(); ++j) {
                this->getData()[i][j] *= scalar;
            }
        }
    }

    // Solve using LU decomposition
    void luDecomposition(Mt& L, Mt& U, Mt& P) {
        int n = this->numRows();

        L.resize(n,n);
        U.resize(n,n);
        P.resize(n,n);

        // Initialize permutation matrix P as identity
        for (int i = 0; i < n; ++i) {
            P.getData()[i][i] = 1.0;
        }

        // Initialize lower triangular matrix L as identity
        for (int i = 0; i < n; ++i) {
            L.getData()[i][i] = 1.0;
        }

        // Pivot selection
        for (int i = 0; i < n; ++i) {
            int maxRow = i;
            for (int k = i + 1; k < n; ++k) {
                if (abs(this->getData()[k][i]) > abs(this->getData()[maxRow][i])) {
                    maxRow = k;
                }
            }

            this->swapRows(i, maxRow);
            P.swapRows(i, maxRow);

            for (int j = i; j < n; ++j) {
                U.getData()[i][j] = this->getData()[i][j];
                for (int k = 0; k < i; ++k) {
                    U.getData()[i][j] -= L.getData()[i][k] * U.getData()[k][j];
                }
            }

            for (int j = i + 1; j < n; ++j) {
                L.getData()[j][i] = this->getData()[j][i];
                for (int k = 0; k < i; ++k) {
                    L.getData()[j][i] -= L.getData()[j][k] * U.getData()[k][i];
                }
                L.getData()[j][i] /= U.getData()[i][i];
            }
        }
    }

    // Solve system using LU decomposition
    Vc solveApproachLU(const Vc b) {
        Mt tempA = *this;
        Vc tempB = b;
        int n = tempA.numRows();
        Mt L, U, P;
        tempA.luDecomposition(L, U, P);

        Vc Pb = P.multiplyMtVc(tempB);
        Vc y(n);
        for (int i = 0; i < n; ++i) {
            y.getCoords(i) = Pb.getCoords(i);
            for (int j = 0; j < i; ++j) {
                y.getCoords(i) -= L.getData()[i][j] * y.getCoords(j);
            }
        }

        Vc x(n);
        for (int i = n - 1; i >= 0; --i) {
            x.getCoords(i) = y.getCoords(i);
            for (int j = i + 1; j < n; ++j) {
                x.getCoords(i) -= U.getData()[i][j] * x.getCoords(j);
            }
            x.getCoords(i) /= U.getData()[i][i];
        }

        return x;
    }

    // QR decomposition using Householder reflections
    void QRDecomposition(Mt& Q, Mt& R) {
        int n = this->numRows();
        Q.resize(n, n);
        R = *this;

        for (int i = 0; i < n; ++i) {
            Q.getData()[i][i] = 1.0;
        }

        for (int i = 0; i < n - 1; ++i) {
            Vc y(n - i);
            for (int j = i; j < n; ++j) {
                y.getCoords(j - i) = R.getData()[j][i];
            }
            Vc z(n - i);
            z.getCoords(0) = 1.0;

            double normY = y.norm();
            Vc w = y - z * normY;
            w.normalize();

            Mt H(n, n);  // Householder matrix
            for (int j = 0; j < n; ++j) {
                for (int k = 0; k < n; ++k) {
                    H.getData()[j][k] = -2 * w.getCoords(j) * w.getCoords(k);
                }
                H.getData()[j][j] += 1;
            }
            R = multiplyMatrices(H, R);
            Mt Qk = multiplyMatrices(Q, H);
            Q = Qk;
        }
    }

    // Solve system using QR decomposition
    Vc solveApproachQR(const Vc b) {
        Mt tempA = *this;
        Vc tempB = b;
        int n = tempA.numRows();
        Mt Q, R;
        tempA.QRDecomposition(Q, R);

        Vc Qt_b(Q.getRowSize(0));
        for (size_t i = 0; i < Q.getRowSize(0); ++i) {
            for (size_t j = 0; j < Q.numRows(); ++j) {
                Qt_b.getCoords(i) += Q.getData()[j][i] * tempB.getCoords(j);
            }
        }

        Vc x(R.getRowSize(0));
        for (int i = R.getRowSize(0) - 1; i >= 0; --i) {
            double sum = 0.0;
            for (size_t j = i + 1; j < R.getRowSize(0); ++j) {
                sum += R.getData()[i][j] * x.getCoords(j);
            }
            x.getCoords(i) = (Qt_b.getCoords(i) - sum) / R.getData()[i][i];
        }

        return x;
    }

    // Solve using Method of Simple Iterations
    Vc solveApproachMPI(Vc& b, double epsilon, int &iter) {
        Mt tempA = *this;
        Vc tempB = b;
        int n = tempA.numRows();
        Vc x(n);
        Vc xPrev(n);

        double normA = tempA.matrixInfNorm();
        double m = 1.0 / normA;

        Mt B(n,n);
        Vc c(n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    B.getData()[i][j] = -m * tempA.getData()[i][j];
                } else {
                    B.getData()[i][j] = 1.0 - m * tempA.getData()[i][j];
                }
            }
            c.getCoords(i) = m * tempB.getCoords(i);
        }

        if (B.matrixInfNorm() >= 1) {
            Mt At = multiplyMatrices(tempA.transposeMt(), tempA);
            Vc bt = (tempA.transposeMt()).multiplyMtVc(tempB);
            m = 1.0 / At.matrixInfNorm();
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (i != j) {
                        B.getData()[i][j] = -m * At.getData()[i][j];
                    } else {
                        B.getData()[i][j] = 1.0 - m * At.getData()[i][j];
                    }
                }
                c.getCoords(i) = m * bt.getCoords(i);
            }
        }

        xPrev = c;
        iter = 0;
        while (true) {
            iter++;
            for (int i = 0; i < n; ++i) {
                double sum = 0.0;
                for (int j = 0; j < n; ++j) {
                    sum += B.getData()[i][j] * xPrev.getCoords(j);
                }
                x.getCoords(i) = sum + c.getCoords(i);
            }

            Vc xd = x - xPrev;
            double error;
            if (B.matrixInfNorm() >= 1) {
                Mt At = multiplyMatrices(tempA.transposeMt(), tempA);
                Vc bt = (tempA.transposeMt()).multiplyMtVc(tempB);
                Vc Atx = At.multiplyMtVc(x);
                Vc bxt = Atx - bt;
                error = bxt.exactNorm();
            }
            else {
                error = (B.matrixInfNorm()/(1 - B.matrixInfNorm()))*xd.exactNorm();
            }

            if (error <= epsilon) break;
            xPrev = x;
        }
        return x;
    }

    // Solve using Gauss-Seidel method
    Vc solveApproachSeidel(Vc&b, double epsilon, int &iter) {
        Mt tempA = *this;
        Vc tempB = b;
        int n = tempA.numRows();
        Vc x(n);

        Mt C(n, n);
        Vc d(n);
        if (tempA.isDiagonalDominant()) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (i != j) {
                        C.getData()[i][j] = -tempA.getData()[i][j] / tempA.getData()[i][i];
                    } else {
                        C.getData()[i][j] = 0;
                    }
                }
                d.getCoords(i) = tempB.getCoords(i) / tempA.getData()[i][i];
            }
        }
        else {
            Mt At = multiplyMatrices(tempA.transposeMt(), tempA);
            Vc bt = (tempA.transposeMt()).multiplyMtVc(tempB);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (i != j) {
                        C.getData()[i][j] = -At.getData()[i][j] / At.getData()[i][i];
                    } else {
                        C.getData()[i][j] = 0;
                    }
                }
                d.getCoords(i) = bt.getCoords(i) / At.getData()[i][i];
            }
        }

        x = d;
        iter = 0;
        while (true) {
            for (int i = 0; i < n; ++i) {
                double sum = 0.0;
                for (int j = 0; j < n; ++j) {
                    sum += C.getData()[i][j] * x.getCoords(j);
                }
                x.getCoords(i) = sum + d.getCoords(i);
            }

            double error;
            if (tempA.isDiagonalDominant()) {
                Vc Ax = tempA.multiplyMtVc(x);
                Vc Axb = Ax - tempB;
                error = Axb.norm();
            }
            else {
                Mt At = multiplyMatrices(tempA.transposeMt(), tempA);
                Vc bt = (tempA.transposeMt()).multiplyMtVc(tempB);
                Vc Atx = At.multiplyMtVc(x);
                Vc bxt = Atx - bt;
                error = bxt.norm();
            }

            if (error <= epsilon) break;
            iter++;
        }

        return x;
    }
};
