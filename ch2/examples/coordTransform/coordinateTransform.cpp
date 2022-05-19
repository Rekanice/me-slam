#include <iostream>
#include <vector>
#include <algorithm>
#include "Eigen/Core"
#include "Eigen/Geometry"

using namespace std;
using namespace Eigen;

int main(int argc, char** argv) {
    Quaterniond q1(0.35, 0.2, 0.3, 0.1), q2(-0.5, 0.4, -0.1, 0.2);
    q1.normalize(); // Note that the quaternion needs to be normalized before use.
    q2.normalize();
    Vector3d t1(0.3, 0.1, 0.1), t2(-0.1, 0.5, 0.3);
    Vector3d p1(0.5, 0, 0.2);   //The test point

    Isometry3d T1w(q1), T2w(q2);  //The rotation matrix part
    T1w.pretranslate (t1);
    T2w.pretranslate (t2);
    
    Vector3d p2 = T2w * T1w.inverse() * p1; //pR2 = T(R2,W)*T(W,R1)*pR1
    cout << endl << p2 << endl;
    return 0;
}