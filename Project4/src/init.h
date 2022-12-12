#include<iostream>
#include <opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <math.h>
#include "matplotlibcpp.h"
using namespace std;
using namespace cv;
using namespace Eigen;

// 归一化匹配点
void Normalize(const vector<KeyPoint> &vKeys, vector<Point2f> &vNormalizedPoints, Matrix3f &T);

// 计算单应性矩阵
Matrix3f ComputeH21(const vector<Point2f> &vP1, const vector<Point2f> &vP2);

// 验证内点数
int CheckHomography(const Matrix3f &H21, const Matrix3f &H12, const vector<KeyPoint> &vP1, const vector<KeyPoint> &vP2, vector<DMatch> &matchers, vector<bool> &vbMatchesInliers);


int CheckHomography_(const Matrix3f &H21, const Matrix3f &H12, const vector<Point2f> &vP1, const vector<Point2f> &vP2, vector<DMatch> &matchers, vector<bool> &vbMatchesInliers);


Matrix3f ComputeF21(const vector<Point2f> &vP1, const vector<Point2f> &vP2);

int CheckFundamental(const Matrix3f &F21, const vector<KeyPoint> &vP1, const vector<KeyPoint> &vP2, vector<DMatch> &matchers, vector<bool> &vbMatchesInliers);

// 通过H矩阵求解R，t和三角化的地图点
bool ReconstructH(vector<DMatch> &vbMatchesInliers, vector<DMatch> &best_match, Matrix3f &K, Matrix3f &H, Matrix3f &R21, Vector3f &t21, const vector<KeyPoint> &vP1, const vector<KeyPoint> &vP2, vector<Point3f> &vP3D);

// 三角化
void Triangulate(const KeyPoint &kp1, const KeyPoint &kp2, const Matrix<float, 3, 4> &P1, const Matrix<float, 3, 4> &P2, Vector3f &x3D);

// 验证投影矩阵
int CheckRT(const Matrix3f &R, const Vector3f &t, const vector<KeyPoint> &vKeys1, const vector<KeyPoint> &vKeys2,
            const vector<DMatch> &vbMatchesInliers,vector<DMatch> &best_match, const Matrix3f &K, vector<Point3f> &vP3D, float th2);

// 计算重投影误差
void pre_err(const Matrix3f &K, const Matrix3f &R1, const Vector3f &t1, const Matrix3f &R2, const Vector3f &t2, double p_w[][3], vector<DMatch> &best_match,const vector<KeyPoint> &vKeys1, const vector<KeyPoint> &vKeys2);


// 重投影显示
vector<KeyPoint> show_err(const Matrix3f &K, const Matrix3f &R, const Vector3f &t, double p_w[][3], int &N);

// 梯度验证
void gradient_check(double &u, double &v, Matrix3d K, const double *const camera, const double *const point);
