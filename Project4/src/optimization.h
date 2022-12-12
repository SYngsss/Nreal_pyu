#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <sophus/se3.hpp>


class SnavelyReprojectionError {
public:
    //传入的是观测值(x,y两个方向)
    SnavelyReprojectionError(float observation_x, float observation_y, Matrix3f K) : observed_x(observation_x), observed_y(observation_y), _K(K){}
    template<typename T>
    bool operator()(const T *const camera,
                    const T *const point,
                    T *residuals) const {
        // camera[0,1,2] are the angle-axis rotation
        T predictions[2];
        // CamProjectionWithDistortion(camera, point, predictions);
        T r[3];
        r[0] = camera[0];
        r[1] = camera[1];
        r[2] = camera[2];

        T p[3];
        ceres::AngleAxisRotatePoint(r, point, p);
        // camera[3,4,5] are the translation
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        T k[4];
        k[0] = T(_K(0,0));
        k[1] = T(_K(1,1));
        k[2] = T(_K(0,2));
        k[3] = T(_K(1,2));

        p[0] = p[0]/p[2];
        p[1] = p[1]/p[2];
        p[2] = p[2]/p[2];

        
        predictions[0] = k[0] * p[0] + k[2];
        predictions[1] = k[1] * p[1] + k[3];

        residuals[0] = predictions[0] - T(observed_x);
        residuals[1] = predictions[1] - T(observed_y);
 
        return true;
    }

 
    static ceres::CostFunction *Create(const float observed_x, const float observed_y, Matrix3f K) {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 6, 3>(
            new SnavelyReprojectionError(observed_x, observed_y, K)));
    }
 
public:
    float observed_x;
    float observed_y;
    Matrix3f _K;
};


class jacobian_error : public ceres::SizedCostFunction<2, 6, 3> {
   public:
     jacobian_error(const double x, const double y, const Matrix3d K) : x_(x), y_(y), K_(K){}
     virtual bool Evaluate(double const* const* parameters,
                           double* residuals,
                           double** jacobians) const {
       const double * cam = parameters[0];       
       const double * point = parameters[1];
       Vector3d p_v;
       p_v << point[0], point[1], point[2];
    //    cout<<"in: "<<cam[0]<<" "<<cam[1]<<" "<<cam[2]<<" "<<cam[3]<<" "<<cam[4]<<" "<<cam[5]<<endl;

        double predictions[2];
        double r[3];
        Vector3d axi;
        axi << cam[0], cam[1], cam[2];
        AngleAxisd rotation_vector(axi.norm(),axi.normalized());
        Matrix3d rotation_matrix;
        rotation_matrix=rotation_vector.matrix();
        r[0] = cam[0];
        r[1] = cam[1];
        r[2] = cam[2];
        p_v = rotation_matrix * p_v ;

        double p[3], x,y,z,x_2, y_2, z_2;
        // ceres::AngleAxisRotatePoint(r, point, p);
        // camera[3,4,5] are the translation
        p[0] = p_v[0] + cam[3];
        p[1] = p_v[1] + cam[4];
        p[2] = p_v[2] + cam[5];

        x = p[0];
        x_2 = x*x;
        y = p[1];
        y_2 = y*y;
        z = p[2];
        z_2 = z*z;

        double k[4];
        k[0] = K_(0,0);
        k[1] = K_(1,1);
        k[2] = K_(0,2);
        k[3] = K_(1,2);

        p[0] = p[0]/p[2];
        p[1] = p[1]/p[2];
        p[2] = p[2]/p[2];
        
        predictions[0] = k[0] * p[0] + k[2];
        predictions[1] = k[1] * p[1] + k[3];

        residuals[0] = x_ - predictions[0];
        residuals[1] = y_ - predictions[1];
        


        if(jacobians != NULL)
        {
            double inv_z = 1./z;
			double inv_z2 = 1./z_2;
            if(jacobians[0] != NULL)
            {
                Map<Matrix<double, 2, 6>> J0(jacobians[0]);
                J0(0,0) = -k[0]*inv_z;
                J0(0,1) = 0;
                J0(0,2) = k[0]*x*inv_z2;
                J0(0,3) = k[0]*x*y*inv_z2;
                J0(0,4) = -k[0] - k[0]*x_2*inv_z2;
                J0(0,5) = k[0]*y*inv_z;

                J0(1,0) = 0;
                J0(1,1) = -k[1]*inv_z;
                J0(1,2) = k[1]*y*inv_z2;
                J0(1,3) = k[1]+k[1]*y_2*inv_z2;
                J0(1,4) = -k[1]*x*y*inv_z2;
                J0(1,5) = -k[1]*x*inv_z;

                for(int i=0;i<6;i++) jacobians[0][i] = J0(0,i);
                for(int j=0;j<6;j++) jacobians[0][j+6] = J0(1,j);
            }

            if(jacobians[1] != NULL)
            {
                Map<Matrix<double, 2, 3>> J1(jacobians[1]);
                J1(0,0) = k[0]*inv_z;
                J1(0,1) = 0;
                J1(0,2) = -k[0]*x*inv_z2;
                J1(1,0) = 0;
                J1(1,1) = k[1]*inv_z;
                J1(1,2) = -k[1]*y*inv_z2;
                J1 = -J1 * rotation_matrix;
                for(int i=0;i<3;i++) jacobians[1][i] = J1(0,i);
                for(int j=0;j<3;j++) jacobians[1][j+3] = J1(1,j);
            }
        }
       return true;
     }

   /*
    Matrix<double, 6, 1> camera_se3(parameters[0]);
		Sophus::SE3d camera_SE3 = Sophus::SE3d::exp(camera_se3);
		Vector3d _point(parameters[1]);
        // cout<<"   ===: "<<_point<<endl;
        
		Vector3d Pc = camera_SE3 * _point;
		Vector2d residual = (K_ * Pc).hnormalized();

		residuals[0] = x_ - residual[0];
		residuals[1] = y_ - residual[1];

		if(jacobians != NULL) {
			double x = Pc[0];
			double y = Pc[1];
			double z = Pc[2];
			double x2 = x*x;
			double y2 = y*y;
			double z2 = z*z;
			double inv_z = 1/z;
			double inv_z2 = 1/z2;
			double fx = K_(0,0);
			double fy = K_(1,1);
			double cx = K_(0,2);
			double cy = K_(1,2);

			if(jacobians[0] != NULL) {
				Map<Matrix<double, 2, 6, RowMajor>> J1(jacobians[0]);
				J1(0,0) =  fx*inv_z;
				J1(0,1) =  0;
				J1(0,2) = -fx*x*inv_z2;
				J1(0,3) = -fx*x*y*inv_z2;
				J1(0,4) =  fx + fx*x2*inv_z2;
				J1(0,5) = -fx*y*inv_z;
				J1(1,0) =  0;
				J1(1,1) =  fy*inv_z;
				J1(1,2) = -fy*y*inv_z2;
				J1(1,3) = -fy - fy*y2*inv_z2;
				J1(1,4) =  fy*x*y*inv_z2;
				J1(1,5) =  fy*x*inv_z;

				J1 = -J1;
			}
			if(jacobians[1] != NULL) {
				Map<Matrix<double, 2, 3, RowMajor>> J2(jacobians[1]);
				J2(0,0) =  fx*inv_z;
				J2(0,1) =  0;
				J2(0,2) = -fx*x*inv_z2;
				J2(1,0) =  0;
				J2(1,1) =  fy*inv_z;
				J2(1,2) = -fy*y*inv_z2;

				J2 = -J2*camera_SE3.rotationMatrix();
			}
		}

		return true;
        
	}
    */
    
    private:
     const double x_;
     const double y_;
     const Matrix3d K_;
};


class Rat43Analytic : public ceres::SizedCostFunction<1,4> {
   public:
     Rat43Analytic(const double x, const double y) : x_(x), y_(y) {}
     virtual ~Rat43Analytic() {}
     virtual bool Evaluate(double const* const* parameters,
                           double* residuals,
                           double** jacobians) const {
       const double b1 = parameters[0][0];
       const double b2 = parameters[0][1];
       const double b3 = parameters[0][2];
       const double b4 = parameters[0][3];

       residuals[0] = b1 *  pow(1 + exp(b2 -  b3 * x_), -1.0 / b4) - y_;

       if (!jacobians) return true;
       double* jacobian = jacobians[0];
       if (!jacobian) return true;

       jacobian[0] = pow(1 + exp(b2 - b3 * x_), -1.0 / b4);
       jacobian[1] = -b1 * exp(b2 - b3 * x_) *
                     pow(1 + exp(b2 - b3 * x_), -1.0 / b4 - 1) / b4;
       jacobian[2] = x_ * b1 * exp(b2 - b3 * x_) *
                     pow(1 + exp(b2 - b3 * x_), -1.0 / b4 - 1) / b4;
       jacobian[3] = b1 * log(1 + exp(b2 - b3 * x_)) *
                     pow(1 + exp(b2 - b3 * x_), -1.0 / b4) / (b4 * b4);
       return true;
     }

    private:
     const double x_;
     const double y_;
 };



