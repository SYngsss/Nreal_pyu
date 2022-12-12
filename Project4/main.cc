#include "init.h"
#include "optimization.h"
#include <opencv2/core/eigen.hpp>
#include <opencv2/xfeatures2d.hpp>
#include<ctime>
#include <chrono>
using namespace std;
using namespace cv;


int main()
{
    Mat img1 = imread("two_image_pose_estimation/1403637188088318976.png",-1);
    Mat img2 = imread("two_image_pose_estimation/1403637189138319104.png",-1);

    const Mat K = ( Mat_<double> ( 3,3 ) << 458.654, 0.0, 367.215, 0.0, 457.296, 248.375, 0.0, 0.0, 1.0 );
    const Mat D = ( Mat_<double> ( 4,1 ) << -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05 );
    Size imageSize(752, 480);
    const double alpha = 0;
    Mat NewCameraMatrix = getOptimalNewCameraMatrix(K, D, imageSize, alpha, imageSize, 0);
    Mat undistort_img1,undistort_img2;
    undistort(img1, undistort_img1, K, D, NewCameraMatrix);
    undistort(img2, undistort_img2, K, D, NewCameraMatrix);

    int numfeature = 1000; //设置特征点个数
    Ptr<xfeatures2d::SURF>Detector = xfeatures2d::SURF::create(numfeature);
    vector<KeyPoint> kp_img1, kp_img2;
	Mat descriptor_img1, descriptor_img2;
    Mat image1,image2;
	Detector->detectAndCompute(undistort_img1, Mat(), kp_img1, descriptor_img1);
    Detector->detectAndCompute(undistort_img2, Mat(), kp_img2, descriptor_img2);

	// // printf("检测到的左图所有的特征点个数：%d\n", kp_img1.size()); //打印检测到的特征点个数
    // drawKeypoints(undistort_img1,kp_img1,image1,Scalar(255,0,255));
    // drawKeypoints(undistort_img2,kp_img2,image2,Scalar(255,0,255));

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
	vector<DMatch> matchers;
	matcher->match(descriptor_img1, descriptor_img2, matchers);

// RANSAC
    const int N = matchers.size();
    const int mMaxIterations = 800; // 最大迭代次数
    
    // 新建一个容器vAllIndices存储特征点索引，并预分配空间
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);

	//在RANSAC的某次迭代中，还可用的索引
    vector<size_t> vAvailableIndices;
	//初始化所有特征点对的索引，索引值0到N-1
    for(int i=0; i<N; i++)
    {
        vAllIndices.push_back(i);
    }

    // 在所有匹配特征点对中随机选择8对匹配特征点为一组
    // 共选择 mMaxIterations组，最大迭代次数
    // mvSets保存每次迭代时所使用的匹配点对
    vector<vector<DMatch>> mvSets = vector< vector<DMatch> >(mMaxIterations,vector<DMatch>(0));
    srand(time(0));
	// 取八对点的匹配
    for(int it=0; it<mMaxIterations; it++)
    {
		// 迭代开始的时候，所有的点都是可用的
        vAvailableIndices = vAllIndices;

        // Select a minimum set
		// 使用八点法求，所以这里就循环了八次
        for(size_t j=0; j<8; j++)
        {
            int nn = vAvailableIndices.size();
            // 随机产生一对点的id,范围从0到N-1
            int randi = (rand() % nn);
            // idx表示哪一个索引对应的特征点对被选中
            DMatch idx = matchers[vAvailableIndices[randi]];
			// cout<<idx.queryIdx<<" ====== "<<idx.trainIdx<<endl;
			// 将本次迭代这个选中的第j个特征点对的索引添加到mvSets中
            mvSets[it].push_back(idx);

            // 由于这对点在本次迭代中已经被使用了,所以我们为了避免再次抽到这个点,就进行删除
            vAvailableIndices[randi] = vAvailableIndices.back();
			vAvailableIndices.pop_back();
        }
    }

    // 归一化后的特征点坐标
    vector<Point2f> vPn1, vPn2;

	// 记录各自的归一化矩阵
    Matrix3f T1, T2;
    Normalize(kp_img1,vPn1,T1);
    Normalize(kp_img2,vPn2,T2);

    // Vector3f a(kp_img1[100].pt.x,kp_img1[100].pt.y,1.0);
    // cout<<vPn1[100]<<endl;
    // cout<<T1*a<<endl;

    Matrix3f T2inv = T2.inverse();
    Matrix3f T2t = T2.transpose();

    // 最佳内点数
    int bestH = 0;
    int best_ = 0;
	// 最佳得分的内点标记
    vector<bool> vbMatchesInliersH = vector<bool>(N,false);

    // 某次迭代的匹配点
    vector<Point2f> vPn1i(8);
    vector<Point2f> vPn2i(8);

    // 单应性矩阵
    Matrix3f H21i, H12i;

    // 最好的单应性矩阵
    Matrix3f H;

    // 当前内点数
    int numH = 0;
    int num_ = 0;
    // 当前的内点数
    vector<bool> vbCurrentInliersH(N,false);

    // 迭代
    for(int it=0; it<mMaxIterations; it++)
    {
        
		// 选择8个归一化之后的点对进行迭代
        for(size_t j=0; j<8; j++)
        {
			// 从mvSets中获取当前次迭代的特征点对
            vector<DMatch> idx = mvSets[it];

            // vPn1i和vPn2i为匹配的特征点对的归一化后的坐标
            vPn1i[j] = vPn1[idx[j].queryIdx];
            vPn2i[j] = vPn2[idx[j].trainIdx];
        }


        // 单应性矩阵计算
        Matrix3f Hn = ComputeH21(vPn1i,vPn2i);

        // 恢复单应性矩阵
        H21i = T2inv*Hn*T1;
		// 计算H逆
        H12i = H21i.inverse();

        numH = CheckHomography(H21i,H12i,kp_img1,kp_img2,matchers,vbCurrentInliersH);
        if(numH > bestH) 
        {
            bestH = numH;
            vbMatchesInliersH = vbCurrentInliersH;
            H = H21i;
        }

    }
    cout<<bestH<<endl;
    cout<<N<<endl;
    cout<<" "<<endl;



    vector<DMatch> mat_H,mat_F;
        for(int i=0;i<N;i++)
        {
            if(vbMatchesInliersH[i]){
                mat_H.push_back(matchers[i]);
            }       
        }
    
	Mat img_matches,img_H,img_F, img_H_tri;
	// drawMatches(undistort_img1, kp_img1, undistort_img2, kp_img2, matchers, img_matches);
    // drawMatches(undistort_img1, kp_img1, undistort_img2, kp_img2, mat_H, img_H);
    // 使用H矩阵恢复位姿
    Matrix3f K_,R21;
    Vector3f t21;
    vector<Point3f> P3D; // 三角化的点
    cv2eigen(K, K_);
    vector<DMatch> best_match;
    ReconstructH(mat_H, best_match, K_, H, R21, t21, kp_img1, kp_img2, P3D);
    // drawMatches(undistort_img1, kp_img1, undistort_img2, kp_img2, best_match, img_H_tri);
    Mat a;
    eigen2cv(R21,a);
    Mat b ;

    Rodrigues(a,b);
    double cam1[6] = {0}, cam2[6];
    cam2[0] = double(b.at<float>(0));
    cam2[1] = double(b.at<float>(1));
    cam2[2] = double(b.at<float>(2));
    cam2[3] = t21[0];
    cam2[4] = t21[1];
    cam2[5] = t21[2];

    double cam1_[6] = {0}, cam2_[6];
// Sophus::SE3d camera1_SE3;
// Sophus::SE3d camera2_SE3(R21, t21);
// Sophus::Vector6d camera1_se3 = camera1_SE3.log();
// Sophus::Vector6d camera2_se3 = camera2_SE3.log();
    cam2_[0] = double(b.at<float>(0));
    cam2_[1] = double(b.at<float>(1));
    cam2_[2] = double(b.at<float>(2));
    cam2_[3] = t21[0];
    cam2_[4] = t21[1];
    cam2_[5] = t21[2];


    Matrix3d kk;
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<3;j++) kk(i,j) = double(K_(i,j)); 
    }

    // 自动求导
    ceres::Problem problem_aoto;
    double p_w[P3D.size()][3];
    for (int i = 0; i < P3D.size(); ++i) 
    {
        // cout<<P3D[i].x<<" "<<P3D[i].y<<" "<<P3D[i].z<<endl;
        p_w[i][0] = P3D[i].x;p_w[i][1] = P3D[i].y;p_w[i][2] = P3D[i].z;

        ceres::CostFunction* cost_function1 =
            SnavelyReprojectionError::Create(
                kp_img1[best_match[i].queryIdx].pt.x,
                kp_img1[best_match[i].queryIdx].pt.y, K_);

        ceres::CostFunction* cost_function2 =
            SnavelyReprojectionError::Create(
                kp_img2[best_match[i].trainIdx].pt.x,
                kp_img2[best_match[i].trainIdx].pt.y, K_);
        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

        problem_aoto.AddResidualBlock(cost_function1,
                                NULL /* squared loss */,
                                cam1,
                                p_w[i]);

         problem_aoto.AddResidualBlock(cost_function2,
                                NULL /* squared loss */,
                                cam2,
                                p_w[i]);
         problem_aoto.SetParameterBlockConstant(cam1);
    }
        
    ceres::Solver::Options options_auto;
    options_auto.linear_solver_type = ceres::DENSE_SCHUR;
    options_auto.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary_auto;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve(options_auto, &problem_aoto, &summary_auto);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "auto solve time cost = " << time_used.count() << " seconds. " << endl;
    cout << summary_auto.FullReport() << endl;


// 手动求导
// ===================================================================================

    ceres::Problem problem_;
    double p_w_[P3D.size()][3];
    for (int i = 0; i < P3D.size(); i++) 
    {
        // cout<<P3D[i].x<<" "<<P3D[i].y<<" "<<P3D[i].z<<endl;
        p_w_[i][0] = P3D[i].x;p_w_[i][1] = P3D[i].y;p_w_[i][2] = P3D[i].z;

        ceres::CostFunction* cost_function1_ = new jacobian_error( kp_img1[best_match[i].queryIdx].pt.x
        , kp_img1[best_match[i].queryIdx].pt.y, kk);

        ceres::CostFunction* cost_function2_ = new jacobian_error( kp_img2[best_match[i].trainIdx].pt.x
        , kp_img2[best_match[i].trainIdx].pt.y, kk);
        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

        // cout<<"waican111: "<<cam1_[0]<<" "<<cam1_[1]<<" "<<cam1_[2]<<" "<<cam1_[3]<<" "<<cam1_[4]<<" "<<cam1_[5]<<endl;
        problem_.AddResidualBlock(cost_function1_,
                                NULL /* squared loss */,
                                cam1_,
                                p_w_[i]);
        
        // cout<<"waican222: "<<cam2_[0]<<" "<<cam2_[1]<<" "<<cam2_[2]<<" "<<cam2_[3]<<" "<<cam2_[4]<<" "<<cam2_[5]<<endl;
        // cout<<endl;
         problem_.AddResidualBlock(cost_function2_,
                                NULL /* squared loss */,
                                cam2_,
                                p_w_[i]);
        problem_.SetParameterBlockConstant(cam1_);
    }
        
    ceres::Solver::Options options_;
    options_.linear_solver_type = ceres::DENSE_SCHUR;
    options_.minimizer_progress_to_stdout = true;
    // options_.check_gradients = true; // 自动求解梯度对比
    ceres::Solver::Summary summary_;
    chrono::steady_clock::time_point t1_ = chrono::steady_clock::now();
    ceres::Solve(options_, &problem_, &summary_);
    chrono::steady_clock::time_point t2_ = chrono::steady_clock::now();
    chrono::duration<double> time_used_ = chrono::duration_cast<chrono::duration<double>>(t2_ - t1_);
    cout << "J solve time cost = " << time_used_.count() << " seconds. " << endl;

    cout << summary_.FullReport() << endl;

    for(int i=0;i<6;i++)
    {
        cout<<"auto1: "<<cam1[i]<<"  J1: "<<cam1_[i]<<endl;
        cout<<"auto2: "<<cam2[i]<<"  J2: "<<cam2_[i]<<endl;
    }

  

    Mat temp_a1(3,1,CV_32F), temp_a2(3,1,CV_32F);
    temp_a1 = (Mat_<float>(3,1)<<cam1[0],cam1[1],cam1[2]);
    temp_a2 = (Mat_<float>(3,1)<<cam2[0],cam2[1],cam2[2]);
    Mat temp_b1, temp_b2 ;
    Rodrigues(temp_a1,temp_b1);
    Rodrigues(temp_a2,temp_b2);
    Matrix3f R_new1, R_new2;
    Vector3f t_new1, t_new2;
    cv2eigen(temp_b2,R_new2);
    cv2eigen(temp_b1,R_new1);
    t_new2[0] = cam2[3];
    t_new2[1] = cam2[4];
    t_new2[2] = cam2[5];
    t_new1[0] = cam1[3];
    t_new1[1] = cam1[4];
    t_new1[2] = cam1[5];

    // 输出误差并绘图
    pre_err(K_, R_new1, t_new1, R_new2, t_new2, p_w, best_match, kp_img1, kp_img2);




    Mat temp_a1_(3,1,CV_32F), temp_a2_(3,1,CV_32F);
    temp_a1_ = (Mat_<float>(3,1)<<cam1_[0],cam1_[1],cam1_[2]);
    temp_a2_ = (Mat_<float>(3,1)<<cam2_[0],cam2_[1],cam2_[2]);
    Mat temp_b1_, temp_b2_ ;
    Rodrigues(temp_a1_,temp_b1_);
    Rodrigues(temp_a2_,temp_b2_);
    Matrix3f R_new1_, R_new2_;
    Vector3f t_new1_, t_new2_;
    cv2eigen(temp_b2_,R_new2_);
    cv2eigen(temp_b1_,R_new1_);
    t_new2_[0] = cam2_[3];
    t_new2_[1] = cam2_[4];
    t_new2_[2] = cam2_[5];
    t_new1_[0] = cam1_[3];
    t_new1_[1] = cam1_[4];
    t_new1_[2] = cam1_[5];

    // 输出误差并绘图
    pre_err(K_, R_new1_, t_new1_, R_new2_, t_new2_, p_w_, best_match, kp_img1, kp_img2);

    



    // 重投影显示
    int n = best_match.size();
    vector<KeyPoint> uv_1, uv_1_bef, uv_2, uv_2_bef;
    uv_2 = show_err(K_, R_new2, t_new2, p_w, n);
    uv_1 = show_err(K_, R_new1, t_new1, p_w, n);

    for(int i=0;i<n;i++)
    {
        KeyPoint *tmp_1 = new KeyPoint;
        KeyPoint *tmp_2 = new KeyPoint;
        tmp_1->pt.x = kp_img1[best_match[i].queryIdx].pt.x;
        tmp_1->pt.y = kp_img1[best_match[i].queryIdx].pt.y;
        tmp_2->pt.x = kp_img2[best_match[i].trainIdx].pt.x;
        tmp_2->pt.y = kp_img2[best_match[i].trainIdx].pt.y;
        uv_2_bef.push_back(*tmp_2);
        uv_1_bef.push_back(*tmp_1);
    }

    drawKeypoints(undistort_img2,uv_2,image2,Scalar(0,0,255));
    drawKeypoints(image2,uv_2_bef,image2,Scalar(255,0,0));

    drawKeypoints(undistort_img1,uv_1,image1,Scalar(0,0,255));
    drawKeypoints(image1,uv_1_bef,image1,Scalar(255,0,0));

    imshow("image1",image1);
    imshow("image2",image2);
    waitKey(0);
    return 0;
}