#include<iostream>
#include<Eigen/Dense>


int main(){
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