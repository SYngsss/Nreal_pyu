#include"init.h"
#include <opencv2/core/eigen.hpp>
#include <opencv2/xfeatures2d.hpp>
#include<ctime>

using namespace std;
using namespace cv;
int main()
{
    Mat img1 = imread("two_image_pose_estimation/1403637188088318976.png",-1);
    Mat img2 = imread("two_image_pose_estimation/1403637189138319104.png",-1);
    // cout<<Range(0,10).start<<"========="<<Range(0,10).end<<endl;

/*===========================2====================================

    // cout<<img1.channels()<<endl;
    Mat emp = Mat::zeros(img1.size(), img1.type());

    for(int i=0;i<img1.rows;i++)
    {
        const uchar* img1_ptr = img1.ptr<uchar>(i);
        uchar* emp_ptr = emp.ptr<uchar>(i);
        
        for(int j=0;j<img1.cols;j++)
        {
            emp_ptr[j] = 255 - img1_ptr[j];
        }
    }
    imshow("emp",emp);
    imshow("a",img1);
    imshow("b",img2);
    imwrite("reverse.png",emp);
    waitKey(0);
    //=================================================================
    // =================================================================
    // =================================================================
*/


/*===========================3====================================
    const Mat K = ( Mat_<double> ( 3,3 ) << 458.654, 0.0, 367.215, 0.0, 457.296, 248.375, 0.0, 0.0, 1.0 );
    const Mat D = ( Mat_<double> ( 4,1 ) << -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05 );
    Size imageSize(752, 480);
    //当alpha=1时，所有像素均保留，但存在黑色边框,保持原来的视野，包含所有像素
    //当alpha=0时，损失最多的像素，没有黑色边框，裁剪，放大图像
    const double alpha = 1;
    Mat NewCameraMatrix = getOptimalNewCameraMatrix(K, D, imageSize, alpha, imageSize, 0); // 根据根据比例因子返回相应的新的相机内参矩阵
    Mat undistort_img1;
    undistort(img1, undistort_img1, K, D, NewCameraMatrix); // 其内部调用了initUndistortRectifyMap和remap函数


    Mat map1, map2;
    // 计算原始图像和矫正图像之间的转换关系 将KDRP转换为map1和map2.
    initUndistortRectifyMap(K, D, Mat(), NewCameraMatrix, imageSize, CV_32FC1, map1, map2);
    Mat undistort_img2;
    remap(img2, undistort_img2, map1, map2, INTER_LINEAR); // 重复图像的话，畸变转换map的对应关系只需要计算一次即可，重映射作用
    imshow("img1",img1);
    imshow("undistort_img1",undistort_img1);
    imshow("img2",img2);
    imshow("undistort_img2",undistort_img2);
    waitKey(0);

        // undistortPoints采用迭代的方式来求解去畸变点：假设无畸变，使用迭代公式
        // x′= (x−2p1 xy−p2 (r^2 + 2x^2))∕( 1 + k1*r^2 + k2*r^4 + k3*r^6)
        // y′= (y−2p2 xy−p1 (r^2 + 2y^2))∕( 1 + k1*r^2 + k2*r^4 + k3*r^6)
        // 迭代求解出来的点再畸变回去，使用原点去计算畸变误差，不满足阈值就继续迭代。

        // remap是直接申请了最大的二维范围32767x32767，通过map1和map2的映射关系，将申请的范围映射到原图中，然后对原图的对应位置进行像素点的取值操作
        
   //================================================================================
   //================================================================================
   //================================================================================
*/



//===================================4===================================
    const Mat K = ( Mat_<double> ( 3,3 ) << 458.654, 0.0, 367.215, 0.0, 457.296, 248.375, 0.0, 0.0, 1.0 );
    const Mat D = ( Mat_<double> ( 4,1 ) << -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05 );
    Size imageSize(752, 480);
    const double alpha = 0;
    Mat NewCameraMatrix = getOptimalNewCameraMatrix(K, D, imageSize, alpha, imageSize, 0);
    Mat undistort_img1,undistort_img2;
    undistort(img1, undistort_img1, K, D, NewCameraMatrix);
    undistort(img2, undistort_img2, K, D, NewCameraMatrix);

    int numfeature = 1000; //设置特征点个数
    Ptr<xfeatures2d::SIFT>Detector = xfeatures2d::SIFT::create(numfeature);
    vector<KeyPoint> kp_img1, kp_img2;
	Mat descriptor_img1, descriptor_img2;
    Mat image1,image2;
    // GaussianBlur(undistort_img1,undistort_img1,Size(5,5),0);
    // GaussianBlur(undistort_img2,undistort_img2,Size(5,5),0);
    // undistort_img1
	Detector->detectAndCompute(undistort_img1, Mat(), kp_img1, descriptor_img1);
    Detector->detectAndCompute(undistort_img2, Mat(), kp_img2, descriptor_img2);

/* 校正畸变匹配点
    Mat kp1(kp_img1.size(),2,CV_32F);
	//遍历每个特征点，并将它们的坐标保存到矩阵中
    for(int i=0; i<kp_img1.size(); i++)
    {
		//然后将这个特征点的横纵坐标分别保存
        kp1.at<float>(i,0)=kp_img1[i].pt.x;
        kp1.at<float>(i,1)=kp_img1[i].pt.y;
    }

    Mat kp2(kp_img2.size(),2,CV_32F);
	//遍历每个特征点，并将它们的坐标保存到矩阵中
    for(int i=0; i<kp_img2.size(); i++)
    {
		//然后将这个特征点的横纵坐标分别保存
        kp2.at<float>(i,0)=kp_img2[i].pt.x;
        kp2.at<float>(i,1)=kp_img2[i].pt.y;
    }

    kp1=kp1.reshape(2);
    kp2=kp2.reshape(2);
    undistortPoints(kp1,kp1,K,D,Mat(),NewCameraMatrix);
    undistortPoints(kp2,kp2,K,D,Mat(),NewCameraMatrix);
    kp1=kp1.reshape(1);
    kp2=kp2.reshape(1);

    for(int i=0; i<kp_img1.size(); i++)
    {
		//根据索引获取这个特征点
		//注意之所以这样做而不是直接重新声明一个特征点对象的目的是，能够得到源特征点对象的其他属性
        cv::KeyPoint kp = kp_img1[i];
		//读取校正后的坐标并覆盖老坐标
        kp.pt.x=kp1.at<float>(i,0);
        kp.pt.y=kp1.at<float>(i,1);
    }

    for(int i=0; i<kp_img2.size(); i++)
    {
		//根据索引获取这个特征点
		//注意之所以这样做而不是直接重新声明一个特征点对象的目的是，能够得到源特征点对象的其他属性
        cv::KeyPoint kp = kp_img2[i];
		//读取校正后的坐标并覆盖老坐标
        kp.pt.x=kp2.at<float>(i,0);
        kp.pt.y=kp2.at<float>(i,1);
    }
*/
    
	// printf("检测到的左图所有的特征点个数：%d\n", kp_img1.size()); //打印检测到的特征点个数
    drawKeypoints(undistort_img1,kp_img1,image1,Scalar(255,0,255));
    drawKeypoints(undistort_img2,kp_img2,image2,Scalar(255,0,255));

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
	vector<DMatch> matchers;
	matcher->match(descriptor_img1, descriptor_img2, matchers);
    
/* 使用阈值来筛选匹配点
    double max_dist = matchers[0].distance;
    double min_dist = matchers[0].distance;
    for(int i=1;i<numfeature;i++)
    {
        double dist = matchers[i].distance;

        if(dist > max_dist) max_dist = dist;
        if(dist < min_dist) min_dist = dist;
    }

    // cout<<max_dist<<endl;
    // cout<<min_dist<<endl;

    vector<DMatch> better_matchers;
    for (int i = 0; i < matchers.size(); ++i)
    {
        double dist = matchers[i].distance;
        if (dist < 2 * min_dist)
        better_matchers.push_back(matchers[i]);
    }
    cout << "goodMatches:" << better_matchers.size() << endl;
*/


// RANSAC
    const int N = matchers.size();
    const int mMaxIterations = 500; // 最大迭代次数
    
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
    int bestF = 0;
	// 最佳得分的内点标记
    vector<bool> vbMatchesInliersH = vector<bool>(N,false);
    vector<bool> vbMatchesInliersF = vector<bool>(N,false);

    // 某次迭代的匹配点
    vector<Point2f> vPn1i(8);
    vector<Point2f> vPn2i(8);

    // 单应性矩阵和基础矩阵
    Matrix3f H21i, H12i,F21i;

    // 最好的单应性矩阵和基础矩阵
    Matrix3f H, F;

    // 当前内点数
    int numH = 0;
    int num_ = 0;
    int numF = 0;
    // 当前的内点数
    vector<bool> vbCurrentInliersF(N,false);
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
        Matrix3f Fn = ComputeF21(vPn1i,vPn2i);
        // num_ = CheckHomography_(Hn,Hn.inverse(),vPn1,vPn2,matchers,vbCurrentInliers);

        // 恢复单应性矩阵
        H21i = T2inv*Hn*T1;
		// 计算H逆
        H12i = H21i.inverse();
        // 恢复基础矩阵
        F21i = T2t*Fn*T1;

        numH = CheckHomography(H21i,H12i,kp_img1,kp_img2,matchers,vbCurrentInliersH);
        numF = CheckFundamental(F21i,kp_img1,kp_img2,matchers,vbCurrentInliersF);
        if(numH > bestH) 
        {
            bestH = numH;
            vbMatchesInliersH = vbCurrentInliersH;
            H = H21i;
        }

        if(numF > bestF) 
        {
            bestF = numF;
            vbMatchesInliersF = vbCurrentInliersF;
            F = F21i;
        }

        // cout<<num<<endl;
    }
    cout<<bestH<<endl;
    cout<<bestF<<endl;
    cout<<N<<endl;
    cout<<" "<<endl;



    vector<DMatch> mat_H,mat_F;
        for(int i=0;i<N;i++)
        {
            if(vbMatchesInliersH[i]){
                mat_H.push_back(matchers[i]);
            }

            if(vbMatchesInliersF[i]){
                mat_F.push_back(matchers[i]);
            }
                
        }
    
	Mat img_matches,img_H,img_F, img_H_tri;
	drawMatches(undistort_img1, kp_img1, undistort_img2, kp_img2, matchers, img_matches);
    drawMatches(undistort_img1, kp_img1, undistort_img2, kp_img2, mat_H, img_H);
    drawMatches(undistort_img1, kp_img1, undistort_img2, kp_img2, mat_F, img_F);

    // 使用H矩阵恢复位姿
    Matrix3f K_,R21;
    Vector3f t21;
    vector<Point3f> P3D; // 三角化的点
    cv2eigen(K, K_);
    vector<DMatch> best_match;
    ReconstructH(mat_H, best_match, K_, H, R21, t21, kp_img1, kp_img2, P3D);

    drawMatches(undistort_img1, kp_img1, undistort_img2, kp_img2, best_match, img_H_tri);


    imshow("image1",image1);
    imshow("image2",image2);
    imshow("img_matches",img_matches);
    imshow("img_H",img_H);
    imshow("img_F",img_F);
    imshow("img_H_tri",img_H_tri);
    
    waitKey(0);
    return 0;
}