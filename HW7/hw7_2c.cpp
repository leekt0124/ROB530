#include <iostream>
#include <fstream>
#include <string>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/utilities.h>



using namespace std;
using namespace gtsam;

struct Pose {
    int i;
    double x = 0, y = 0, z = 0;
    Quaternion q; 
    Pose(int i_, double x_, double y_, double z_, Quaternion q_) : i(i_), x(x_), y(y_), z(z_), q(q_) {}
    // friend ostream& operator<<(ostream& os, const Pose& dt);
};

struct Edge {
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW 
    int i, j;
    double x, y, z;
    Quaternion q;
    Eigen::Matrix<double, 6, 6, Eigen::DontAlign> m;
    Edge(int i_, int j_, double x_, double y_, double z_, Quaternion q_, Eigen::Matrix<double, 6, 6, Eigen::DontAlign> m_) :
    i(i_), j(j_), x(x_), y(y_), z(z_), q(q_), m(m_) {} 
    friend ostream& operator<<(ostream& os, const Edge& dt);

    
};

// ostream& operator<<(ostream& os, const Pose& dt)
// {
//     os << dt.i << " " << dt.x << " " << dt.y << " " << dt.theta;
//     return os;
// }

ostream& operator<<(ostream& os, const Edge& dt)
{
    os << dt.i << " " << dt.x << " " << dt.y << " " << dt.m << " " << dt.z << endl;
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

    auto priorNoise = noiseModel::Diagonal::Sigmas((Vector(6)<<0.3,0.3,0.3,0.1,0.1,0.1).finished());

    ifstream fin("/home/leekt/UMich/ROB530/HW7/data/parking-garage.g2o");
    while (!fin.eof()) {
        string name;
        fin >> name;
        cout << name << endl;

        if (name == "VERTEX_SE3:QUAT") {
            int i = 0;
            fin >> i;
            double x = 0, y = 0, z = 0;
            double qx, qy, qz, qw;
            fin >> x >> y >> z >> qx >> qy >> qz >> qw;
            Quaternion q(qw, qx, qy, qz);
            poses.push_back(Pose(i, x, y, z, q));
        }

        else if (name == "EDGE_SE3:QUAT") {
            int i = 0, j = 0;
            double x = 0, y = 0, z = 0, qx = 0, qy = 0, qz = 0, qw = 0; 
            double q11, q12, q13, q14, q15, q16, q22, q23, q24, q25, q26, q33, q34, q35, q36, q44, q45, q46, q55, q56, q66;
            fin >> i >> j >> x >> y >> z >> qx >> qy >> qz >> qw >> q11 >> q12 >> q13 >> q14 >> q15 >> q16 >> q22 >> q23 >> q24 >> q25 >> q26 >> q33 >> q34 >> q35 >> q36 >> q44 >> q45 >> q46 >> q55 >> q56 >> q66;
            Quaternion q(qw, qx, qy, qz);
            Eigen::Matrix<double, 6, 6, Eigen::DontAlign> m;
            m << q11, q12, q13, q14, q15, q16, 
                 q12, q22, q23, q24, q25, q26, 
                 q13, q23, q33, q34, q35, q36, 
                 q14, q24, q34, q44, q45, q46, 
                 q15, q25, q35, q45, q55, q56, 
                 q16, q26, q36, q46, q56, q66;
            edges.push_back(Edge(i, j, x, y, z, q, m));
        }

        if (!fin.good()) break;
    }


    NonlinearFactorGraph graph;
    Values initialEstimate;

    for (auto& pose : poses) {
        int i = pose.i;
        double x = pose.x, y = pose.y, z = pose.z;
        Quaternion q = pose.q;
        if (i == 0) {
            graph.addPrior(i, Pose3(q, {x, y, z}), priorNoise);
            initialEstimate.insert(i, Pose3(q, {x, y, z}));
        }
        else {
            auto prevPose = result.at<Pose3>(i - 1);
            initialEstimate.insert(i, prevPose);
            for (auto& edge : edges) {
                if (edge.j == pose.i) {
                    // Eigen::Matrix<double, 6, 6, Eigen::DontAlign> m = edge.m;
                    auto model = noiseModel::Gaussian::Information(edge.m);
                    graph.emplace_shared<BetweenFactor<Pose3>> (edge.i, edge.j, Pose3(edge.q, {edge.x, edge.y, edge.z}), model);
                }
            }
        }
        isam.update(graph, initialEstimate);
        result = isam.calculateEstimate();
        graph.resize(0);
        initialEstimate.clear();
    }

    // std::cout << "initial error=" << graph->error(*initial) << std::endl;
    std::cout << "final error=" << graph.error(result) << std::endl;

    result.print("Final Result:\n"); // This step will print final values

    cout << "Saving data to txt file..." << endl;
    std::ofstream optimized_file("/home/leekt/UMich/ROB530/HW7/plot/2_c_optimized.txt");
    // std::ofstream initial_file("/home/leekt/UMich/ROB530/HW7/plot/1_cinitial.txt");
    for (int i = 0; i < result.size(); ++i) {
        float x = result.at<Pose3>(i).x();
        float y = result.at<Pose3>(i).y();
        float z = result.at<Pose3>(i).z();
        // float theta = result.at<Pose3>(i).theta();
        string s = to_string(x) + " " + to_string(y) + " " + to_string(z) + "\n";
        optimized_file << s;

        // float i_x = initialEstimate.at<Pose2>(i).x();
        // float i_y = initialEstimate.at<Pose2>(i).y();
        // float i_theta = initialEstimate.at<Pose2>(i).theta();
        // string i_s = to_string(i_x) + " " + to_string(i_y) + " " + to_string(i_theta) + "\n";
        // initial_file << i_s;
    }

    return 0;
}