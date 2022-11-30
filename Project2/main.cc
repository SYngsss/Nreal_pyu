#include<iostream>
#include <Eigen/Dense>
#include<fstream>
#include<Eigen/QR>
using namespace std;
using namespace Eigen;
void test_1()
{
	Eigen::Quaterniond imu_q_cam_left(
		0.99090224973327068,
		0.13431639597354814,
		0.00095051670014565813,
		-0.0084222184858180373);

	Eigen::Vector3d imu_p_cam_left(
		-0.050720060477640147,
		-0.0017414170413474165,
		0.0022943667597148118);

	Eigen::Quaterniond imu_q_cam_right(
		0.99073762672679389,
		0.13492462817073628,
		-0.00013648999867379373,
		-0.015306242884176362);

	Eigen::Vector3d imu_p_cam_rigrt(
		0.051932496584961352,
		-0.0011555929083120534,
		0.0030949732069645722);


	Eigen::Matrix3d R_left_cam;//声明一个Eigen类的3*3的旋转矩阵
    //四元数转为旋转矩阵--先归一化再转为旋转矩阵
    R_left_cam = imu_q_cam_left.normalized().toRotationMatrix();

	Eigen::Matrix3d R_right_cam;
    R_right_cam = imu_q_cam_right.normalized().toRotationMatrix();

	// std::cout<<R_left_cam<<std::endl;
	// std::cout<<R_right_cam<<std::endl;


	// R = R'T * R''  t = R'T * t'' - R'T * t'
	Eigen::Matrix3d R;
	Eigen::Vector3d p;

	R = R_left_cam.transpose() * R_right_cam;
	p = R_left_cam.transpose() * (imu_p_cam_rigrt - imu_p_cam_left);

	Eigen::Quaterniond imu_q_cam_left_text = imu_q_cam_left.normalized().conjugate();
	Eigen::Quaterniond result = imu_q_cam_left_text * imu_q_cam_right.normalized();

	Eigen::Quaterniond q(R);
	std::cout<<"R: "<<q.x()<<" "<<q.y()<<" "<<q.z()<<" "<<q.w()<<std::endl;
	std::cout<<"result: "<<result.x()<<" "<<result.y()<<" "<<result.z()<<" "<<result.w()<<std::endl;
	std::cout<<"t: "<<p<<std::endl;
}

void test_3()
{
	ifstream fdata;
	string strDataPath = "data.txt";
    fdata.open(strDataPath);
	MatrixXf A(100,2);
	VectorXf b(100);
    string s1;
    getline(fdata,s1);
    // 遍历文件 end时为真，反之为假
	for(int i=0;i<100;i++)
	{
		getline(fdata,s1);
		stringstream ss1;
        ss1 << s1;
		float x,y;
		ss1 >> x;
		ss1 >> y;
		A(i,0) = x;
		A(i,1) = 1;
		b(i) = y;
	}
	// cout<<"A: "<<A<<endl;
	// cout<<"b: "<<b<<endl;
	Vector2f re_ldl,re_qr,re_svd,re_lu;

	// 这种方法通常是最快的，尤其是当A “又高又瘦”的时候。
	// 但是，即使矩阵A有轻微病态，这也不是一个好方法，
	// 因为A T A的条件数是A的条件数的平方。这意味着与上面提到的更稳定的方法相比，使用正规方程式会损失大约两倍的精度。
	re_ldl = (A.transpose() * A).ldlt().solve(A.transpose()* b);
	// Eigen中QR分解类有三个：HouseholderQR（没有枢转，快速但不稳定）
	// ColPivHouseholderQR（列枢转，因此有一些慢但是更准确）
	// FullPivHouseholderQR（完全旋转，最慢，但是最稳定）
	re_qr = A.householderQr().solve(b); 

	// LU需要可逆的条件
	re_lu = A.fullPivLu().solve(b);

	// SVD分解通常准确率最高，但是速度最慢
	re_svd = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b);

	cout<<"re_ldl "<<re_ldl<<endl;
	cout<<"re_qr "<<re_qr<<endl;
	cout<<"re_lu "<<re_lu<<endl;
	cout<<"re_svd "<<re_svd<<endl;

}

void test_4()
{
	// https://zhuanlan.zhihu.com/p/91393594?ivk_sa=1024320u
	// 条件数越大，越接近奇异矩阵(不可逆) 矩阵越病态
	ifstream fdata;
	string strDataPath = "data.txt";
    fdata.open(strDataPath);
	MatrixXd A(100,2);
	VectorXd b(100);
    string s1;
    getline(fdata,s1);
    // 遍历文件 end时为真，反之为假
	for(int i=0;i<100;i++)
	{
		getline(fdata,s1);
		stringstream ss1;
        ss1 << s1;
		double x,y;
		ss1 >> x;
		ss1 >> y;
		A(i,0) = x;
		A(i,1) = 1;
		b(i) = y;
	}



	ifstream fdata2;
	string strDataPath2 = "data2.txt";
    fdata2.open(strDataPath2);
	MatrixXd A2(99,2);
	VectorXd b2(99);
    string s2;
    getline(fdata2,s2);
    // 遍历文件 end时为真，反之为假
	for(int i=0;i<99;i++)
	{
		getline(fdata2,s2);
		stringstream ss2;
        ss2 << s2;
		double x,y;
		ss2 >> x;
		ss2 >> y;
		A2(i,0) = x;
		A2(i,1) = 1;
		b2(i) = y;
	}
	
	JacobiSVD<MatrixXd> re_svd1(A,ComputeFullU | ComputeFullU);
	JacobiSVD<MatrixXd> re_svd2(A2,ComputeFullU | ComputeFullU);

	cout<<"svd_1条件数 "<<re_svd1.singularValues()<<endl;
	cout<<"svd_2条件数 "<<re_svd2.singularValues()<<endl;
	cout<<"svd_1条件数 "<<re_svd1.singularValues()[0]/re_svd1.singularValues()[1]<<endl;
	cout<<"svd_2条件数 "<<re_svd2.singularValues()[0]/re_svd2.singularValues()[1]<<endl;


}



int main(){
	// test_1();
	test_4();
}