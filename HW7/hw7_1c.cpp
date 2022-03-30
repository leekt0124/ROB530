#include <iostream>
#include <fstream>
#include <string>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/ISAM2.h>


using namespace std;
using namespace gtsam;

struct Pose {
    int i;
    double x, y, theta;
    Pose(int i_, double x_, double y_, double theta_) : i(i_), x(x_), y(y_), theta(theta_) {}
    friend ostream& operator<<(ostream& os, const Pose& dt);
};

struct Edge {
    int i, j;
    double x, y, theta;
    Eigen::Matrix<double, 3, 3> info;
    Edge(int i_, int j_, double x_, double y_, double theta_, Eigen::Matrix<double, 3, 3> info_) :
    i(i_), j(j_), x(x_), y(y_), theta(theta_), info(info_) {} 
    friend ostream& operator<<(ostream& os, const Edge& dt);
};

ostream& operator<<(ostream& os, const Pose& dt)
{
    os << dt.i << " " << dt.x << " " << dt.y << " " << dt.theta;
    return os;
}

ostream& operator<<(ostream& os, const Edge& dt)
{
    os << dt.i << " " << dt.x << " " << dt.y << " " << dt.theta << " " << dt.info;
    return os;
}

int main() {

    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    ISAM2 isam(parameters);
    Values result = isam.calculateEstimate();

    vector<Pose> poses;    
    vector<Edge> edges;

    auto priorNoise = noiseModel::Diagonal::Sigmas(Vector3(0.3, 0.3, 0.1));

    fstream newfile;
    newfile.open("/home/leekt/UMich/ROB530/HW7/input_INTEL_g2o.g2o", ios::in);
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
                poses.push_back(Pose(i, x, y, theta));
                
            }

            else if (v[0] == "EDGE_SE2") {
                int i = stoi(v[1]), j = stoi(v[2]);
                double x = stod(v[3]), y = stod(v[4]), theta = stod(v[5]), q11 = stod(v[6]), q12 = stod(v[7]), q13 = stod(v[8]), q22 = stod(v[9]), q23 = stod(v[10]), q33 = stod(v[11]);
                Eigen::Matrix<double, 3, 3> info;
                info << q11, q12, q13, q12, q22, q23, q13, q23, q33;
                edges.push_back(Edge(i, j, x, y, theta, info));
            }


        }
        newfile.close();
    }

    // for (auto& pose : poses) {
    //     cout << pose << endl;
    // }
    // for (auto& edge : edges) {
    //     cout << edge << endl;
    // }

    NonlinearFactorGraph graph;
    Values initialEstimate;

    for (auto& pose : poses) {
        int i = pose.i;
        double x = pose.x, y = pose.y, theta = pose.theta;
        if (i == 0) {
            graph.addPrior(i, Pose2(x, y, theta), priorNoise);
            initialEstimate.insert(i, Pose2(x, y, theta));
        }
        else {
            auto prevPose = result.at<Pose2>(i - 1);
            initialEstimate.insert(i, prevPose);
            for (auto& edge : edges) {
                if (edge.j == pose.i) {
                    auto model = noiseModel::Gaussian::Information(edge.info);
                    graph.emplace_shared<BetweenFactor<Pose2>> (edge.i, edge.j, Pose2(edge.x, edge.y, edge.theta), model);
                }
            }
        }
        isam.update(graph, initialEstimate);
        result = isam.calculateEstimate();
        graph.resize(0);
        initialEstimate.clear();
    }
    
    result.print("Final Result:\n"); // This step will print final values

    cout << "Saving data to txt file..." << endl;
    std::ofstream optimized_file("/home/leekt/UMich/ROB530/HW7/plot/1_c_optimized.txt");
    // std::ofstream initial_file("/home/leekt/UMich/ROB530/HW7/plot/1_cinitial.txt");
    for (int i = 0; i < result.size(); ++i) {
        float x = result.at<Pose2>(i).x();
        float y = result.at<Pose2>(i).y();
        float theta = result.at<Pose2>(i).theta();
        string s = to_string(x) + " " + to_string(y) + " " + to_string(theta) + "\n";
        optimized_file << s;

        // float i_x = initialEstimate.at<Pose2>(i).x();
        // float i_y = initialEstimate.at<Pose2>(i).y();
        // float i_theta = initialEstimate.at<Pose2>(i).theta();
        // string i_s = to_string(i_x) + " " + to_string(i_y) + " " + to_string(i_theta) + "\n";
        // initial_file << i_s;
    }

    return 0;
}