#ifndef MATRIx_H
#define MATRIx_H
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <memory>
namespace ML {

enum MatType{
    NONE = 0,
    ZERO,
    IDENTITY,
    UNIFORM_RAND
};
template<typename T>
class Mat
{
public:
    int rows;
    int cols;
    std::vector<std::vector<T> > data;
public:
	Mat():rows(0), cols(0){}
    ~Mat(){}
    inline bool isSizeEqual(const Mat<T>& x) {return (rows == x.rows && cols == x.cols);}
    inline bool isNull(){return rows == 0 || cols == 0;}
    inline bool isSquare(){return rows == cols;}
    inline T& at(int row, int col){return data[row][col];}
    std::vector<T>& operator[](int i){return data[i];}
    Mat<T>& create(int rows, int cols)
    {
        this->rows = rows;
        this->cols = cols;
        this->data = std::vector<std::vector<T> >(rows);
        for (int i = 0; i < rows; i++) {
            data[i] = std::vector<T>(cols, 0);
        }
        return *this;
    }

    void assign(const Mat<T>& x)
    {
        for (int i = 0; i < rows; i++) {
            data[i] = x.data[i];
        }
        return;
    }

    void assign(T x)
    {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = x;
            }
        }
        return;
    }

    Mat(int rows, int cols)
    {
        create(rows, cols);
        assign(0);
    }

    Mat(const Mat<T>& x)
    {
        if (this == &x) {
            return;
        }
        create(x.rows, x.cols);
        assign(x);
    }

    Mat<T> operator = (const Mat<T>& x)
    {
        if (this == &x) {
            return *this;
        }
        if (isNull()) {
            create(x.rows, x.cols);
        }
        if (!isSizeEqual(x)) {
            std::cout<<"= size not equal"<<std::endl;
            return *this;
        }
        assign(x);
        return *this;
    }

    void zero(){ assign(0);}

    void identity()
    {
        if (isSquare()) {
            return;
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (i == j) {
                    data[i][j] = 1;
                } else {
                    data[i][j] = 0;
                }
            }
        }
        return;
    }

    void random(int n)
    {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = T(rand() % n);
            }
        }
        return;
    }

    void uniformRandom()
    {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = T(rand() % 10000 - rand() % 10000) / 10000;
            }
        }
        return;
    }

    std::vector<T> column(int col)
    {
        std::vector<T> columnT;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (j == col) {
                    columnT.push_back(data[i][j]);
                }
            }
        }
        return columnT;
    }

    std::vector<T> toVector()
    {
        std::vector<T> x;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                x.push_back(data[i][j]);
            }
        }
        return x;
    }

    void show()
    {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                std::cout<<data[i][j]<<" ";
            }
            std::cout<<std::endl;
        }
        return;
    }

    Mat<T> operator + (const Mat<T>& x)
    {
        if (!isSizeEqual(x)) {
            std::cout<<"+ size not equal"<<std::endl;
            return *this;
        }
        Mat<T> y(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                y.data[i][j] = data[i][j] + x.data[i][j];
            }
        }
        return y;
    }

    Mat<T> operator - (const Mat<T>& x)
    {
        if (!isSizeEqual(x)) {
            std::cout<<"- size not equal"<<std::endl;
            return *this;
        }
        Mat<T> y(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                y.data[i][j] = data[i][j] - x.data[i][j];
            }
        }
        return y;
    }

    Mat<T> operator * (const Mat<T>& x)
    {
        if (cols != x.rows) {
            std::cout<<"* size not equal"<<std::endl;
            return *this;
        }
        /* (m, p) x (p, n) = (m, n) */
        int m = rows;
        int p = cols;
        int n = x.cols;
        Mat<T> y(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < p; k++) {
                    y.data[i][j] += data[i][k] * x.data[k][j];
                }
            }
        }
        return y;
    }

    Mat<T> operator / (const Mat<T>& x)
    {
        if (!isSizeEqual(x)) {
            std::cout<<"/ size not equal"<<std::endl;
            return *this;
        }
        Mat<T> y(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                 y.data[i][j] = data[i][j] / x.data[i][j];
            }
        }
        return y;
    }

    Mat<T> operator % (const Mat<T>& x)
    {
        if (!isSizeEqual(x)) {
            std::cout<<"% size not equal"<<std::endl;
            return *this;
        }
        Mat<T> y(rows, cols);
        for (int i = 0; i < y.rows; i++) {
            for (int j = 0; j < y.cols; j++) {
                y.data[i][j] = data[i][j] * x.data[i][j];
            }
        }
        return y;
    }

    Mat<T>& operator += (const Mat<T>& x)
    {
        if (!isSizeEqual(x)) {
            std::cout<<"+= size not equal"<<std::endl;
            return *this;
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] += x.data[i][j];
            }
        }
        return *this;
    }

    Mat<T>& operator -= (const Mat<T>& x)
    {
        if (!isSizeEqual(x)) {
            std::cout<<"-= size not equal"<<std::endl;
            return *this;
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] -= x.data[i][j];
            }
        }
        return *this;
    }

    Mat<T>& operator /= (const Mat<T>& x)
    {
        if (!isSizeEqual(x)) {
            std::cout<<"/= size not equal"<<std::endl;
            return *this;
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] /= x.data[i][j];
            }
        }
        return *this;
    }

    Mat<T>& operator %= (const Mat<T>& x)
    {
        if (!isSizeEqual(x)) {
            std::cout<<"%= size not equal"<<std::endl;
            return *this;
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] *= x.data[i][j];
            }
        }
        return *this;
    }

    Mat<T> operator + (T x)
    {
        Mat<T> y(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                y.data[i][j] = data[i][j] + x;
            }
        }
        return y;
    }

    Mat<T> operator - (T x)
    {
        Mat<T> y(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                y.data[i][j] = data[i][j] - x;
            }
        }
        return y;
    }

    Mat<T> operator * (T x)
    {
        Mat<T> y(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                y.data[i][j] = data[i][j] * x;
            }
        }
        return y;
    }

    Mat<T> operator / (T x)
    {
        Mat<T> y(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                y.data[i][j] = data[i][j] / x;
            }
        }
        return y;
    }

    Mat<T>& operator += (T x)
    {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] += x;
            }
        }
        return *this;
    }

    Mat<T>& operator -= (T x)
    {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] -= x;
            }
        }
        return *this;
    }

    Mat<T>& operator *= (T x)
    {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] *= x;
            }
        }
        return *this;
    }

    Mat<T>& operator /= (T x)
    {
        if (x == 0) {
            return *this;
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] /= x;
            }
        }
        return *this;
    }

    T mean(Mat<T>& x)
    {
        T s = 0;
        T n = 0;
        for (int i = 0; i < x.rows; i++) {
            for (int j = 0; j < x.cols; j++) {
                s += x.data[i][j];
                n++;
            }
        }
        s = s / n;
        return s;
    }

    Mat<T> Tr()
    {
        Mat<T> y(cols,rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                y.data[j][i] = data[i][j];
            }
        }
        return y;
    }

    void save(const std::string& fileName)
    {
        std::ofstream file;
        file.open(fileName, std::ofstream::app);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                file << data[i][j]<<" ";
            }
            file << std::endl;
        }
        return;
    }

    void load(const std::string& fileName)
    {
        std::ifstream file;
        file.open(fileName);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                file >> data[i][j];
            }
        }
        return;
    }
};
template<typename T>
Mat<T> for_each(const Mat<T>& x, double (*func)(double x))
{
    Mat<T> y(x.rows, x.cols);
    for (int i = 0; i < x.rows; i++) {
        for (int j = 0; j < x.cols; j++) {
            y.data[i][j] = func(x.data[i][j]);
        }
    }
    return y;
}

template<typename T>
T sum(Mat<T>& x)
{
    T s = 0;
    for (int i = 0; i < x.rows; i++) {
        for (int j = 0; j < x.cols; j++) {
            s += x.data[i][j];
        }
    }
    return s;
}

template<typename T>
T max(Mat<T>& x)
{
    T maxT = x.data[0][0];
    for (int i = 0; i < x.rows; i++) {
        for (int j = 0; j < x.cols; j++) {
            if (maxT < x.data[i][j]) {
                maxT = x.data[i][j];
            }
        }
    }
    return maxT;
}

template<typename T>
T min(Mat<T>& x)
{
    T minT = x.data[0][0];
    for (int i = 0; i < x.rows; i++) {
        for (int j = 0; j < x.cols; j++) {
            if (minT > x.data[i][j]) {
                minT = x.data[i][j];
            }
        }
    }
    return minT;
}
}
#endif // MATRIx_H
