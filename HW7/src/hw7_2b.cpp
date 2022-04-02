#include <iostream>
#include <fstream>
#include <string>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/utilities.h>

using namespace std;
using namespace gtsam;
// using namespace Eigen;

int main() {
    NonlinearFactorGraph graph;
    Values initialEstimate;

    // auto priorNoise = noiseModel::Diagonal::Sigmas(Vector6(0.3, 0.3, 0.3, 0.1, 0.1, 0.1));
    auto priorNoise = noiseModel::Diagonal::Sigmas((Vector(6)<<0.3,0.3,0.3,0.1,0.1,0.1).finished());

    ifstream fin("../../data/parking-garage.g2o");
    istream& is = fin;

    while (!is.eof()) {
        string name;
        is >> name;

        if (name == "VERTEX_SE3:QUAT") {
            int i = 0;
            is >> i;
            double x = 0, y = 0, z = 0, qx = 0, qy = 0, qz = 0, qw = 0; 
            is >> x >> y >> z >> qx >> qy >> qz >> qw;
            Quaternion q(qw, qx, qy, qz);
            if (i == 0) {
                graph.addPrior(i, Pose3(q, {x, y, z}), priorNoise);
            }
            initialEstimate.insert(i, Pose3(q, {x, y, z}));

        }

        else if (name == "EDGE_SE3:QUAT") {
            int i = 0, j = 0;
            double x = 0, y = 0, z = 0, qx = 0, qy = 0, qz = 0, qw = 0; 
            double q11, q12, q13, q14, q15, q16, q22, q23, q24, q25, q26, q33, q34, q35, q36, q44, q45, q46, q55, q56, q66;
            is >> i >> j >> x >> y >> z >> qx >> qy >> qz >> qw >> q11 >> q12 >> q13 >> q14 >> q15 >> q16 >> q22 >> q23 >> q24 >> q25 >> q26 >> q33 >> q34 >> q35 >> q36 >> q44 >> q45 >> q46 >> q55 >> q56 >> q66;
            Quaternion q(qw, qx, qy, qz);
            Matrix6 m;
            m << q11, q12, q13, q14, q15, q16, 
                 q12, q22, q23, q24, q25, q26, 
                 q13, q23, q33, q34, q35, q36, 
                 q14, q24, q34, q44, q45, q46, 
                 q15, q25, q35, q45, q55, q56, 
                 q16, q26, q36, q46, q56, q66;

            auto model = noiseModel::Gaussian::Information(m);
            graph.emplace_shared<BetweenFactor<Pose3>> (i, j, Pose3(q, {x, y, z}), model);
        }

        if (!is.good()) break;
    }

    utilities::perturbPose2(initialEstimate, 0.05, 0.05);

    GaussNewtonParams parameters;
    // Stop iterating once the change in error between steps is less than this value
    parameters.relativeErrorTol = 1e-5;
    // Do not perform more than N iteration steps
    parameters.maxIterations = 100;
    GaussNewtonOptimizer optimizer(graph, initialEstimate);
    // LevenbergMarquardtOptimizer optimizer(graph, initialEstimate);

    Values result = optimizer.optimize();
    result.print("Final Result:\n"); // This step will print final values

    std::cout << "initial error=" << graph.error(initialEstimate) << std::endl;
    std::cout << "final error=" << graph.error(result) << std::endl;


    cout << "Saving data to txt file..." << endl;
    std::ofstream optimized_file("../../plot/2_b_optimized.txt");
    std::ofstream initial_file("../../plot/2_initial.txt");
    for (int i = 0; i < result.size(); ++i) {
        float x = result.at<Pose3>(i).x();
        float y = result.at<Pose3>(i).y();
        float z = result.at<Pose3>(i).z();
        // float theta = result.at<Pose3>(i).theta();
        string s = to_string(x) + " " + to_string(y) + " " + to_string(z) + "\n";
        optimized_file << s;

        float i_x = initialEstimate.at<Pose3>(i).x();
        float i_y = initialEstimate.at<Pose3>(i).y();
        float i_z = initialEstimate.at<Pose3>(i).z();
        // float theta = result.at<Pose3>(i).theta();
        string i_s = to_string(i_x) + " " + to_string(i_y) + " " + to_string(i_z) + "\n";
        initial_file << i_s;
    }

    return 0;
}