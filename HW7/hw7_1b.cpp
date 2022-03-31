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

    auto priorNoise = noiseModel::Diagonal::Sigmas(Vector3(0.3, 0.3, 0.1));
    // auto odometryNoise = noiseModel::Diagonal::Sigmas(Vector3(0.2, 0.2, 0.1));
    // auto measurementNoise = noiseModel::Diagonal::Sigmas(Vector3(0.1, 0.2));

    fstream newfile;
    newfile.open("/home/leekt/UMich/ROB530/HW7/data/input_INTEL_g2o.g2o", ios::in);
    if (newfile.is_open()) {
        string s;
        while (getline(newfile, s)) {
            // cout << s << endl;
            vector<string> v;
            string temp = "";
            for (int i = 0; i < s.size(); ++i) {
                if (s[i] == ' ') {
                    v.push_back(temp);
                    temp = "";
                }
                else {
                    temp.push_back(s[i]);
                }
            }
            v.push_back(temp);

            if (v[0] == "VERTEX_SE2") {
                int i = stoi(v[1]);
                double x = stod(v[2]), y = stod(v[3]), theta = stod(v[4]);
                // cout << i << " " << x << " " <<  y + z << endl;
                if (i == 0) {
                    graph.addPrior(i, Pose2(x, y, theta), priorNoise);
                }
                initialEstimate.insert(i, Pose2(x, y, theta));
            }

            else if (v[0] == "EDGE_SE2") {
                int i = stoi(v[1]), j = stoi(v[2]);
                double x = stod(v[3]), y = stod(v[4]), theta = stod(v[5]), q11 = stod(v[6]), q12 = stod(v[7]), q13 = stod(v[8]), q22 = stod(v[9]), q23 = stod(v[10]), q33 = stod(v[11]);
                Eigen::Matrix<double, 3, 3> info;
                info << q11, q12, q13, q12, q22, q23, q13, q23, q33;
                auto model = noiseModel::Gaussian::Information(info);
                graph.emplace_shared<BetweenFactor<Pose2>> (i, j, Pose2(x, y, theta), model);
            }


        }
        newfile.close();
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

    cout << "result.dim() = " << result.dim() << endl;

    cout << "Saving data to txt file..." << endl;
    std::ofstream optimized_file("/home/leekt/UMich/ROB530/HW7/plot/1_b_optimized.txt");
    std::ofstream initial_file("/home/leekt/UMich/ROB530/HW7/plot/initial.txt");
    for (int i = 0; i < result.size(); ++i) {
        float x = result.at<Pose2>(i).x();
        float y = result.at<Pose2>(i).y();
        float theta = result.at<Pose2>(i).theta();
        string s = to_string(x) + " " + to_string(y) + " " + to_string(theta) + "\n";
        optimized_file << s;

        float i_x = initialEstimate.at<Pose2>(i).x();
        float i_y = initialEstimate.at<Pose2>(i).y();
        float i_theta = initialEstimate.at<Pose2>(i).theta();
        string i_s = to_string(i_x) + " " + to_string(i_y) + " " + to_string(i_theta) + "\n";
        initial_file << i_s;
    }

    return 0;
}