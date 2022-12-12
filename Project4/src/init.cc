#include"init.h"
namespace plt=matplotlibcpp;
void Normalize(const vector<KeyPoint> &vKeys, vector<Point2f> &vNormalizedPoints, Matrix3f &T)                           //将特征点归一化的矩阵
{
    float meanX = 0;
    float meanY = 0;

    const int N = vKeys.size();

    vNormalizedPoints.resize(N);

	//开始遍历所有的特征点
    for(int i=0; i<N; i++)
    {
        meanX += vKeys[i].pt.x;
        meanY += vKeys[i].pt.y;
    }

    //计算均值
    meanX = meanX/N;
    meanY = meanY/N;

    float meanDevX = 0;
    float meanDevY = 0;

    // 均值设置为0
    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
        vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

		// 累计这些特征点偏离横纵坐标均值的程度
        meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }

    // 平均偏移程度
    meanDevX = meanDevX/N;
    meanDevY = meanDevY/N;
    float sX = 1.0/meanDevX;
    float sY = 1.0/meanDevY;

    // 平均偏移程度归一化到1
    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }

    // |sX  0  -meanx*sX| |x|
    // |0   sY -meany*sY| |y|
    // |0   0      1    | |1|
    T = Matrix3f::Identity();
    T(0,0) = sX;
    T(1,1) = sY;
    T(0,2) = -meanX*sX;
    T(1,2) = -meanY*sY;
}


Matrix3f ComputeH21(const vector<Point2f> &vP1, const vector<Point2f> &vP2)
{
    const int N = vP1.size();

    MatrixXf A(2*N, 9); 

    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A(2*i,0) = 0.0;
        A(2*i,1) = 0.0;
        A(2*i,2) = 0.0;
        A(2*i,3) = -u1;
        A(2*i,4) = -v1;
        A(2*i,5) = -1;
        A(2*i,6) = v2*u1;
        A(2*i,7) = v2*v1;
        A(2*i,8) = v2;

        A(2*i+1,0) = u1;
        A(2*i+1,1) = v1;
        A(2*i+1,2) = 1;
        A(2*i+1,3) = 0.0;
        A(2*i+1,4) = 0.0;
        A(2*i+1,5) = 0.0;
        A(2*i+1,6) = -u2*u1;
        A(2*i+1,7) = -u2*v1;
        A(2*i+1,8) = -u2;
    }

    JacobiSVD<MatrixXf> svd(A, ComputeFullU | ComputeFullV);
    VectorXf f = svd.matrixV().col(8);
    Matrix3f v = Map<Matrix3f>(f.data());
    
    // cout<<v<<endl;
    // cout<<" "<<endl;
    // VectorXf dd(svd.singularValues());
    // int len = dd.size();
    // dd[len-1] = 0;
    // cout<<dd[len-1]<<endl;
    // cout<<"  "<<endl;
    
    return v;
}


int CheckHomography(const Matrix3f &H21, const Matrix3f &H12, const vector<KeyPoint> &vP1, const vector<KeyPoint> &vP2, vector<DMatch> &matchers, vector<bool> &vbMatchesInliers)                       
{
    const int N = matchers.size();
    const float th  = 100;
    int num = 0;

	// 给Inliers标记预分配空间
    vbMatchesInliers.resize(N);

    for(int i = 0; i < N; i++)
    {

        bool bIn = true;

        const Point2f &kp1 = vP1[matchers[i].queryIdx].pt;
        const Point2f &kp2 = vP2[matchers[i].trainIdx].pt;
        Vector3f p1(kp1.x, kp1.y, 1.0);
        Vector3f p2(kp2.x, kp2.y, 1.0);
        
        Vector3f p1_ = H12 * p2;
        // 归一化
        p1_[0] = p1_[0]/p1_[2];
        p1_[1] = p1_[1]/p1_[2];
   
        // 计算重投影误差
        float squareDist1 = (p1[0] - p1_[0]) * (p1[0] - p1_[0]) + (p1[1] - p1_[1]) * (p1[1] - p1_[1]);
        squareDist1 = sqrt(squareDist1);
        // cout<<"squareDist1 "<<squareDist1<<endl;

        if(squareDist1>th)
            bIn = false;    

        Vector3f p2_ = H21 * p1;
        // 归一化
        p2_[0] = p2_[0]/p2_[2];
        p2_[1] = p2_[1]/p2_[2];
   
        // 计算重投影误差
        float squareDist2 = (p2[0] - p2_[0]) * (p2[0] - p2_[0]) + (p2[1] - p2_[1]) * (p2[1] - p2_[1]);
        squareDist2 = sqrt(squareDist2);

        if(squareDist2>th)
            bIn = false;

        if(bIn)
        {
            vbMatchesInliers[i]=true;
            num++;
        }

        else
            vbMatchesInliers[i]=false;
    }
    return num;
}


int CheckHomography_(const Matrix3f &H21, const Matrix3f &H12, const vector<Point2f> &vP1, const vector<Point2f> &vP2, vector<DMatch> &matchers, vector<bool> &vbMatchesInliers)                       
{
    const int N = vP1.size();
    const float th  = 10.;
    int num = 0;

	// 给Inliers标记预分配空间
    vbMatchesInliers.resize(N);

    for(int i = 0; i < N; i++)
    {

        bool bIn = true;

        const Point2f &kp1 = vP1[i];
        const Point2f &kp2 = vP2[i];
        Vector3f p1(kp1.x, kp1.y, 1.0);
        Vector3f p2(kp2.x, kp2.y, 1.0);
        
        Vector3f p1_ = H12 * p2;
        // 归一化
        p1_[0] = p1_[0]/p1_[2];
        p1_[1] = p1_[1]/p1_[2];
   
        // 计算重投影误差
        const float squareDist1 = (p1[0] - p1_[0]) * (p1[0] - p1_[0]) + (p1[1] - p1_[1]) * (p1[1] - p1_[1]);

        // cout<<"squareDist1 "<<squareDist1<<endl;

        if(squareDist1>th)
            bIn = false;    

        Vector3f p2_ = H21 * p1;
        // 归一化
        p2_[0] = p2_[0]/p2_[2];
        p2_[1] = p2_[1]/p2_[2];
   
        // 计算重投影误差
        const float squareDist2 = (p2[0] - p2_[0]) * (p2[0] - p2_[0]) + (p2[1] - p2_[1]) * (p2[1] - p2_[1]);


        if(squareDist2>th)
            bIn = false;

        if(bIn)
        {
            vbMatchesInliers[i]=true;
            num++;
        }

        else
            vbMatchesInliers[i]=false;
    }
    return num;
}


Matrix3f ComputeF21(const vector<Point2f> &vP1, const vector<Point2f> &vP2)
{
   const int N = vP1.size();

    MatrixXf A(2*N, 9); 

    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A(i,0) = u2*u1;
        A(i,1) = u2*v1;
        A(i,2) = u2;
        A(i,3) = v2*u1;
        A(i,4) = v2*v1;
        A(i,5) = v2;
        A(i,6) = u1;
        A(i,7) = v1;
        A(i,8) = 1;
    }

    JacobiSVD<MatrixXf> svd(A, ComputeFullU | ComputeFullV);
    VectorXf f = svd.matrixV().col(8);
    Matrix3f v = Map<Matrix3f>(f.data());

    JacobiSVD<Matrix3f> svd_(v, ComputeFullU | ComputeFullV);
    Vector3f temp = svd_.singularValues();
    int len = temp.size();
    temp[len-1] = 0;
    // VectorXf f = svd.matrixV().col(8);
    // Matrix3f v = Map<Matrix3f>(f.data());
     
    return  svd_.matrixU() * temp.asDiagonal() * svd_.matrixV().transpose();
}


int CheckFundamental(const Matrix3f &F21, const vector<KeyPoint> &vP1, const vector<KeyPoint> &vP2, vector<DMatch> &matchers, vector<bool> &vbMatchesInliers)
{
    const int N = matchers.size();
    int num=0;

    vbMatchesInliers.resize(N);

	
    // const float th = 3.841;
    const float th = 50;

    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        const Point2f &kp1 = vP1[matchers[i].queryIdx].pt;
        const Point2f &kp2 = vP2[matchers[i].trainIdx].pt;

        
        Vector3f p1(kp1.x, kp1.y, 1.0);
        Vector3f p2(kp2.x, kp2.y, 1.0);
        
        Vector3f p1_ = F21 * p1;

    
        // 计算误差 e = (a * p2.x + b * p2.y + c) /  sqrt(a * a + b * b)
        const float num2 = p1_.dot(p2);
        const float squareDist1 = num2*num2/(p1_[0]*p1_[0] + p1_[1]*p1_[1]);

        if(squareDist1>th)
            bIn = false;
       

        Vector3f p2_ = F21.transpose() * p2;

        // 计算误差 e = (a * p2.x + b * p2.y + c) /  sqrt(a * a + b * b)
        const float num1 = p2_.dot(p1);
        const float squareDist2 = num1*num1/(p2_[0]*p2_[0] + p2_[1]*p2_[1]);

        if(squareDist2>th)
            bIn = false;

        if(bIn){
            vbMatchesInliers[i]=true;
            num++;
        }
        else
            vbMatchesInliers[i]=false;
    }
    return num;
}


bool ReconstructH(vector<DMatch> &vbMatchesInliers, vector<DMatch> &best_match, Matrix3f &K, Matrix3f &H, Matrix3f &R21, Vector3f &t21, const vector<KeyPoint> &vP1, const vector<KeyPoint> &vP2, vector<Point3f> &vP3D)
{
    Matrix3f invK = K.inverse();
    Matrix3f H21 = invK*H*K;
    int N=vbMatchesInliers.size();
    JacobiSVD<Matrix3f> svd(H21, ComputeFullU | ComputeFullV);
    float s = svd.matrixU().determinant() * svd.matrixV().transpose().determinant();

    float d1 = svd.singularValues()[0];
    float d2 = svd.singularValues()[1];
    float d3 = svd.singularValues()[2];

    if(d1/d2<1.00001 || d2/d3<1.00001) {
        return false;
    }

    vector<Matrix3f> vR;
    vector<Vector3f> vt;
    vR.reserve(8);
    vt.reserve(8);

    //  d' > 0 时的 4 组解
    // x1 = e1 * sqrt((d1 * d1 - d2 * d2) / (d1 * d1 - d3 * d3))
    // x2 = 0
    // x3 = e3 * sqrt((d2 * d2 - d2 * d2) / (d1 * d1 - d3 * d3))
    // 令 aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3))
    //    aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3))
    // 则
    // x1 = e1 * aux1
    // x3 = e3 * aux2

    // 因为 e1,e2,e3 = 1 or -1
    // 所以有x1和x3有四种组合
    // x1 =  {aux1,aux1,-aux1,-aux1}
    // x3 =  {aux3,-aux3,aux3,-aux3}
    float aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));
    float aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));
    float x1[] = {aux1,aux1,-aux1,-aux1};
    float x3[] = {aux3,-aux3,aux3,-aux3};


    // sin(theta) = e1 * e3 * sqrt(( d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) /(d1 + d3)/d2
    // cos(theta) = (d2* d2 + d1 * d3) / (d1 + d3) / d2 
    float aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);
    float ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
    float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

    // 计算旋转矩阵 R'
    //根据不同的e1 e3组合所得出来的四种R t的解
    //      | ctheta      0   -aux_stheta|       | aux1|
    // Rp = |    0        1       0      |  tp = |  0  |
    //      | aux_stheta  0    ctheta    |       |-aux3|

    //      | ctheta      0    aux_stheta|       | aux1|
    // Rp = |    0        1       0      |  tp = |  0  |
    //      |-aux_stheta  0    ctheta    |       | aux3|

    //      | ctheta      0    aux_stheta|       |-aux1|
    // Rp = |    0        1       0      |  tp = |  0  |
    //      |-aux_stheta  0    ctheta    |       |-aux3|

    //      | ctheta      0   -aux_stheta|       |-aux1|
    // Rp = |    0        1       0      |  tp = |  0  |
    //      | aux_stheta  0    ctheta    |       | aux3|
    // 开始遍历这四种情况中的每一种
    for(int i=0; i<4; i++)
    {
        
        Matrix3f Rp=Matrix3f::Identity();
        Rp(0,0)=ctheta;
        Rp(0,2)=-stheta[i];
        Rp(2,0)=stheta[i];        
        Rp(2,2)=ctheta;

        Matrix3f R = s*svd.matrixU()*Rp*svd.matrixV().transpose();

        // 保存
        vR.push_back(R);

        Vector3f tp;
        tp(0)=x1[i];
        tp(1)=0;
        tp(2)=-x3[i];
        tp*=d1-d3;

        Vector3f t = svd.matrixU()*tp;
        vt.push_back(t/t.norm());

    }
    
    // 讨论 d' < 0 时的 4 组解
    float aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);
    // cos_theta项
    float cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
    // 考虑到e1,e2的取值，这里的sin_theta有两种可能的解
    float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

    // 四种解的情况
    for(int i=0; i<4; i++)
    {

        Matrix3f Rp=Matrix3f::Identity();
        Rp(0,0)=cphi;
        Rp(0,2)=sphi[i];
        Rp(1,1)=-1;
        Rp(2,0)=sphi[i];
        Rp(2,2)=-cphi;

        Matrix3f R = s*svd.matrixU()*Rp*svd.matrixV().transpose();

        // 保存
        vR.push_back(R);

        Vector3f tp;
        tp(0)=x1[i];
        tp(1)=0;
        tp(2)=x3[i];
        tp*=d1+d3;

        Vector3f t = svd.matrixU()*tp;
        vt.push_back(t/t.norm());
    }

    // 最好的good点
    int bestGood = 0;
    
    // 对 8 组解进行验证，并选择产生相机前方最多3D点的解为最优解
    for(size_t i=0; i<8; i++)
    {
        // 三角化测量之后的特征点的空间坐标
        vector<Point3f> vP3Di;
        vector<DMatch> mati;
    
        int nGood = CheckRT(vR[i],vt[i],vP1,vP2,vbMatchesInliers,mati,K,vP3Di,10);

        if(nGood>bestGood)
        {
            // 更新历史最优点
            bestGood = nGood;
            // 更新变量
            vP3D = vP3Di;
            best_match = mati;
            R21 = vR[i];
            t21 = vt[i];

        }
    }
    cout<<bestGood<<endl;
}


int CheckRT(const Matrix3f &R, const Vector3f &t, const vector<KeyPoint> &vKeys1, const vector<KeyPoint> &vKeys2,
            const vector<DMatch> &vbMatchesInliers,vector<DMatch> &best_match, const Matrix3f &K, vector<Point3f> &vP3D, float th2)
{
    
	//从相机内参数矩阵获取相机的校正参数
    const float fx = K(0,0);
    const float fy = K(1,1);
    const float cx = K(0,2);
    const float cy = K(1,2);

    Matrix<float, 3, 4> P1 = Matrix<float, 3, 4>::Zero();
	P1.col(0) = K.col(0);
    P1.col(1) = K.col(1);
    P1.col(2) = K.col(2);

    Matrix<float, 3, 4> P2 = Matrix<float, 3, 4>::Zero();
    Matrix3f K_R = K*R;
    Vector3f K_t = K*t;
	P2.col(0) = K_R.col(0);
    P2.col(1) = K_R.col(1);
    P2.col(2) = K_R.col(2);
    P2.col(3) = K_t;


    int nGood=0;

	// 开始遍历所有的特征点对
    for(int i=0, iend=vbMatchesInliers.size();i<iend;i++)
    {

        // kp1和kp2是匹配好的有效特征点
        const KeyPoint &kp1 = vKeys1[vbMatchesInliers[i].queryIdx];
        const KeyPoint &kp2 = vKeys2[vbMatchesInliers[i].trainIdx];
		//存储三维点的的坐标
        Vector3f p3dC1, p3dC2;

        // 利用三角法恢复三维点p3dC1
        Triangulate(kp1,kp2,P1,P2,p3dC1);
    
        if(!isfinite(p3dC1(0)) || !isfinite(p3dC1(1)) || !isfinite(p3dC1(2)) || p3dC1(2)<=0) continue;
        
        p3dC2 = R*p3dC1+t;	
        if(p3dC2(2)<=0) continue;
        
        float im1x, im1y;
        float invZ1 = 1.0/p3dC1(2);
        im1x = fx*p3dC1(0)*invZ1+cx;
        im1y = fy*p3dC1(1)*invZ1+cy;

		//参考帧上的重投影误差
        float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);
        squareError1 = sqrt(squareError1);
        if(squareError1>th2)
            continue;

        float im2x, im2y;
        float invZ2 = 1.0/p3dC2(2);
        im2x = fx*p3dC2(0)*invZ2+cx;
        im2y = fy*p3dC2(1)*invZ2+cy;

		// 计算重投影误差
        float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);
        squareError2 = sqrt(squareError2);
        if(squareError2>th2)
            continue;

        vP3D.push_back(Point3f(p3dC1(0),p3dC1(1),p3dC1(2)));
        best_match.push_back(vbMatchesInliers[i]);
        nGood++;
    }
    
    return nGood;
}


void Triangulate(const KeyPoint &kp1, const KeyPoint &kp2, const Matrix<float, 3, 4> &P1, const Matrix<float, 3, 4> &P2, Vector3f &x3D)
{
    Matrix4f A;

	//构造参数矩阵A
    A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
    A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
    A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
    A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

    JacobiSVD<Matrix4f> svd(A, ComputeFullU | ComputeFullV);
    Vector4f a3D = svd.matrixV().col(3);
    a3D = a3D/a3D[3];
    x3D = a3D.head(3);
}

void pre_err(const Matrix3f &K, const Matrix3f &R1, const Vector3f &t1, const Matrix3f &R2, const Vector3f &t2, double p_w[][3], vector<DMatch> &best_match,const vector<KeyPoint> &vKeys1, const vector<KeyPoint> &vKeys2)
{

    // 三维点个数
    int N = best_match.size();

    // 每个三维点的误差
    vector<double> err_1, err_x1, err_y1;
    vector<double> err_2, err_x2, err_y2;
    // double err[N], err_x[N], err_y[N]; 

    // 总误差
    double sum_err1 = 0;
    double sum_err2 = 0;

    // 特征点坐标
    float u1,v1,u2,v2;

    // 误差均值、方差、标准差
    double err_mean1 = 0, err_variance1 = 0, err_variance_stan1 = 0;
    double err_mean2 = 0, err_variance2 = 0, err_variance_stan2 = 0;

    for (int i = 0; i < N; ++i) 
    {
        u1 = vKeys1[best_match[i].queryIdx].pt.x;
        v1 = vKeys1[best_match[i].queryIdx].pt.y;

        u2 = vKeys2[best_match[i].trainIdx].pt.x;
        v2 = vKeys2[best_match[i].trainIdx].pt.y;

        Vector3f p1,p2;
        p1 << p_w[i][0],p_w[i][1],p_w[i][2];
        p2 << p_w[i][0],p_w[i][1],p_w[i][2];

        p1 = R1*p1 + t1;
        p1[0] = p1[0]/p1[2];p1[1] = p1[1]/p1[2];p1[2] = 1.0;
        p2 = R2*p2 + t2;
        p2[0] = p2[0]/p2[2];p2[1] = p2[1]/p2[2];p2[2] = 1.0;
        p2 = K * p2;
        p1 = K * p1;

        err_x1.push_back(p1[0] - u1);
        err_y1.push_back(p1[1] - v1);
        err_x2.push_back(p2[0] - u2);
        err_y2.push_back(p2[1] - v2);

        err_1.push_back(sqrt((p1[0] - u1) * (p1[0] - u1) + (p1[1] - v1) * (p1[1] - v1)));
        err_2.push_back(sqrt((p2[0] - u2) * (p2[0] - u2) + (p2[1] - v2) * (p2[1] - v2)));

        sum_err1 += err_1[i];
        sum_err2 += err_2[i];
    }

    // 均值
    err_mean1 = sum_err1/N;
    err_mean2 = sum_err2/N;

    for (int i = 0; i < N; ++i) 
    {
        err_variance1 += (err_1[i] - err_mean1) * (err_1[i] - err_mean1);
        err_variance2 += (err_2[i] - err_mean2) * (err_2[i] - err_mean2);
    }
    err_variance1 /= N;
    err_variance_stan1 = sqrt(err_variance1);
    err_variance2 /= N;
    err_variance_stan2 = sqrt(err_variance2);
    cout<<"图一平均误差： "<<err_mean1<<"  图二平均误差： "<<err_mean2<<endl;
    cout<<"图一方差： "<<err_variance1<<"  图二方差： "<<err_variance2<<endl;
    cout<<"图一标准差： "<<err_variance_stan1<<"  图二标准差： "<<err_variance_stan2<<endl;
    // cout<<"平均误差： "<<err_mean2<<endl;
    // cout<<"方差： "<<err_variance2<<endl;
    // cout<<"标准差： "<<err_variance_stan2<<endl;
    // cout<<err_x2.size()<<endl;

    //可视化
    plt::plot(err_x1, err_y1, "bx");
    plt::plot(err_x2, err_y2, "rx");
    plt::show();
}

vector<KeyPoint> show_err(const Matrix3f &K, const Matrix3f &R, const Vector3f &t, double p_w[][3], int &N)
{
    vector<KeyPoint> P_2;
    
    for(int i=0;i<N;i++)
    {
        Vector3f P3D; P3D << p_w[i][0],p_w[i][1],p_w[i][2];
        P3D = K * (R * P3D + t);
        KeyPoint *temp = new KeyPoint;
        temp->pt.x = P3D[0]/P3D[2];
        temp->pt.y = P3D[1]/P3D[2];
        P_2.push_back(*temp);
    }

    return P_2;
}

void gradient_check(double &u, double &v, Matrix3d K, const double *const camera, const double *const point) {


    const double * cam = camera;       
    const double * point_ = point;
    Vector3d p_v;
    double residuals[2];
    p_v << point_[0], point_[1], point_[2];
    double predictions[2];
    Vector3d axi;
    axi << cam[0], cam[1], cam[2];
    AngleAxisd rotation_vector(axi.norm(),axi.normalized());
    Matrix3d rotation_matrix;
    rotation_matrix=rotation_vector.matrix();
    p_v = rotation_matrix * p_v ;

    double p[3], x,y,z,x_2, y_2, z_2;
    p[0] = p_v[0] + cam[3];
    p[1] = p_v[1] + cam[4];
    p[2] = p_v[2] + cam[5];

    x = p[0];
    x_2 = x*x;
    y = p[1];
    y_2 = y*y;
    z = p[2];
    z_2 = z*z;
    double inv_z = 1./z;
	double inv_z2 = 1./z_2;

    double k[4];
    k[0] = K(0,0);
    k[1] = K(1,1);
    k[2] = K(0,2);
    k[3] = K(1,2);

    p[0] = p[0]/p[2];
    p[1] = p[1]/p[2];
    p[2] = p[2]/p[2];
    
    predictions[0] = k[0] * p[0] + k[2];
    predictions[1] = k[1] * p[1] + k[3];

    residuals[0] = u - predictions[0];
    residuals[1] = v - predictions[1];
    

    Matrix<double, 2, 6> J0;
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
    // 计算的原本雅可比
	cout << J0 << endl;

	double err = 1e-7;
	Matrix<double, 2, 6> J0_;
    Matrix<double, 6, 1> cam_(camera);
    Vector3d p_v_;
    p_v_ << point_[0], point_[1], point_[2];
	for(int i = 0; i < 6; ++i) 
    {
        cam_(i, 0) += err;
        double residuals_[2];
        double predictions_[2];
        Vector3d axi_;
        axi_ << cam_[0], cam_[1], cam_[2];
        AngleAxisd rotation_vector_(axi_.norm(),axi_.normalized());
        Matrix3d rotation_matrix_;
        rotation_matrix_=rotation_vector_.matrix();
        p_v_ = rotation_matrix_ * p_v_ ;

        double p_[3];
        p_[0] = p_v_[0] + cam_[3];
        p_[1] = p_v_[1] + cam_[4];
        p_[2] = p_v_[2] + cam_[5];


        p_[0] = p_[0]/p_[2];
        p_[1] = p_[1]/p_[2];
        p_[2] = p_[2]/p_[2];
    
        predictions_[0] = k[0] * p_[0] + k[2];
        predictions_[1] = k[1] * p_[1] + k[3];

        residuals_[0] = u - predictions_[0];
        residuals_[1] = v - predictions_[1];
        double err_res[2];
		err_res[0] = residuals_[0] - residuals[0];
        err_res[1] = residuals_[1] - residuals[1];

		J0_(0,i) = err_res[0]/err;
		J0_(1,i) = err_res[1]/err;
	}
	
	// 误差增长量
	cout << J0_ << endl;
}
