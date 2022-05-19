#include <opencv2/opencv.hpp>
#include <string>

using namespace std;

string image_file = "./distorted.png";  // the distorted image

int main(int argc, char **argv) {
    // In this program, we implement the undistortion by ourselves rather than using opencv
    
    // rad-tan model params
    double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;
    // camera intrinsics
    double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;

    cv::Mat image = cv::imread(image_file, 0); // the image type is CV_8UC1
    int rows = image.rows, cols = image.cols;
    cv::Mat image_undistort = cv::Mat(rows, cols, CV_8UC1); // the undistorted image (declared only, no values yet)

    // compute the pixels in the undistorted one
    for (int v = 0; v < rows; v++) {
        for (int u = 0; u < cols; u++) {
            // note we are computing the PIXEL coordinate of (u,v) in the undistorted image
            // according to the rad-tan model, 
            // 1. compute the WORLD coordinates (x', y') for the distorted pixel (u,v) in the distorted image
            double x = (u - cx) / fx, y = (v - cy) / fy;
            double r = sqrt(x * x + y * y);
            double x_distorted = x * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
            double y_distorted = y * (1 + k1 * r * r + k2 * r * r * r * r) + p1 * (r * r + 2 * y * y) + 2 * p2 * x * y;
            // 2. compute the distorted pixel for the distored world coordinate
            double u_distorted = fx * x_distorted + cx;
            double v_distorted = fy * y_distorted + cy;

            // check if the distorted pixel is in the image border
            if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < cols && v_distorted < rows) {
                // if not a border pixel, can fill it at the exact location in the image_undistort matrix declared earlier
                image_undistort.at<uchar>(v, u) = image.at<uchar>((int) v_distorted, (int) u_distorted);  // wah! even need to pecify data type of parameters in cpp
            } else {
                image_undistort.at<uchar>(v, u) = 0;
            }
        }
    }

    // show the undistorted image
    cv::imshow("distorted", image);
    cv::imshow("undistorted", image_undistort);
    cv::waitKey();
    return 0;    
}