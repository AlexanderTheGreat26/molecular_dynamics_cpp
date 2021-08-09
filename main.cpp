/* Program of 3d-modeling gas behavior in unit volume via molecular dynamics method.
 * Interaction of particles modeling via Lennard-Jones potential.
 * Coordinates and velocities defines via Verlet integration.
 * Initial coordinates and velocities directions distributed evenly.
 * By vector in comments it should be understood std::vector<std::tuple<>>,
 * by coordinates, velocities and accelerations it should be understood std::tuple contains projections on coordinate
 * axes.
 * Graphics realised via gnuplot.
 * */

#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <utility>
#include <fstream>
#include <string>
#include <tuple>
#include <array>
#include <algorithm>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <sstream>
#include "omp.h"


// Fundamental constants
const double k_B = 1.380649e-16; // Boltzmann constant, erg/K


// Substance characteristics
const std::string substance = "Ar";
const double m = 6.6335e-23;  // Argon mass, g
const double eps = 119.8;  // Potential pit depth (Theta/k_B), K
const double Theta = k_B * eps;  // Equilibrium temperature
const double sigma = 3.405e-8;  // Smth like shielding length, cm
const double R_0 = sigma * pow(2.0, 1.0/6.0);
const double R_Ar = 0.71e-8;


// Gas characteristics
const double P = 1.0e6;
const double T = 300; // Temperature, K
const double n = P / k_B / T; // Concentration, 1/cm^-3
const double V_init = sqrt(3 * k_B * T / m); // rms speed corresponding to a given temperature (T), cm/c


// Program constants
const int N = 1e2; //Number of particles
const double dt = 1.0e-12; // Time-step
const double simulation_time = 1.0e-7;
const double R_max = 3.0*R_0;
bool realtime = false;


// Model constants
const double Volume = N/n;
const double characteristic_size = std::pow(Volume, 1.0/3.0);
const double left_border = -characteristic_size / 2.0;
const double right_border = characteristic_size / 2.0;


typedef std::tuple<double, double, double> coord;

std::vector<coord> neighboring_cubes; // Contains coordinates of centers virtual areas

std::vector<coord> areas_centers (double a);


typedef std::tuple<double, double, double, double> planes;

std::vector<planes> area_borders = {std::make_tuple(0, 0, 1, left_border),
                                    std::make_tuple(0, 0, 1, right_border),
                                    std::make_tuple(0, 1, 0, left_border),
                                    std::make_tuple(0, 1, 0, right_border),
                                    std::make_tuple(1, 0, 0, left_border),
                                    std::make_tuple(1, 0, 0, right_border)};


//std::random_device rd;  // Will be used to obtain a seed for the random number engine
//std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()


// Initial distributions
std::vector<coord> initial_coordinates ();

std::vector<coord> initial_velocities ();

std::vector<coord> initial_coordinates_debug ();

std::vector<coord> initial_velocities_debug ();



double single_force (double& R);

coord minimal_distance (coord& q1, coord& q2, double& R_ij);

std::vector<coord> total_particle_acceleration (std::vector<coord>& particles);

double cos_ab (coord& a, coord& b);

void momentum_exchange (std::vector<coord>& coordinates, std::vector<coord>& velocities);

void Verlet_integration (std::vector<coord>& q, std::vector<coord>& v);

void Verlet_integration_debug (std::vector<coord>& q, std::vector<coord>& v, std::string& accreleration_files_name);

double kinetic_energy_of_system (std::vector<coord>& velocities);


// Technical functions.
bool is_equal (double a, double b);

bool is_same (std::vector<coord>& a_1, std::vector<coord>& a_2);

void data_file (std::string data_type, std::vector<coord>& data, double& t);

void data_files (std::string& name, std::vector<coord>& data, double& t);

void data_files_debug (std::string& name, std::vector<coord>& data, double& t_debug, bool& flag);

void clear_data (std::string& file_name);

std::string exec (std::string str);


void plot (std::string name, double min, double max, int number_of_points, int& steps);



template<size_t Is = 0, typename... Tp>
void debug_tuple_output (std::tuple<Tp...>& t) {
    std::cout << std::endl;
    std::cout << std::get<Is>(t) << '\t';
    if constexpr(Is + 1 != sizeof...(Tp))
    debug_tuple_output<Is + 1>(t);
}

typedef std::tuple<std::string, std::string, std::string> striiing;



void real_time_plotting (std::vector<coord>& coordinates, std::vector<coord>& velocities,
                         striiing name, double min, double max, int number_of_points, double& E_init);

void computing (striiing &name, std::vector<coord>& coordinates, std::vector<coord>& velocities,
                double& E, double& t);


int main () {
    std::vector<coord> coordinates = std::move(initial_coordinates_debug());
    std::vector<coord> velocities = std::move(initial_velocities_debug());
    neighboring_cubes = std::move(areas_centers(characteristic_size));
    double E, E_init = N * m*std::pow(V_init, 2)/2.0;
    double t = 0;
    std::string name_trajectory = substance + "_coordinates";
    std::string name_velocity = substance + "_velocity";
    std::string name_acceleration = substance + "_acceleration";
    if (!realtime) {
        std::string path_trajectory = std::move(exec("rm -rf trajectories && mkdir trajectories && cd trajectories && echo $PWD"));
        std::string path_velocity = std::move(exec("rm -rf velocities && mkdir velocities && cd velocities && echo $PWD"));
        std::string path_acceleration = std::move(exec("rm -rf accelerations && mkdir accelerations && cd accelerations && echo $PWD"));
        name_trajectory = path_trajectory + '/' + name_trajectory;
        name_velocity = path_velocity + '/' + name_velocity;
        name_acceleration = path_acceleration + '/' + name_acceleration;
        //auto name = std::make_tuple(name_trajectory, name_velocity, name_acceleration);
    }
    striiing name = std::make_tuple(name_trajectory, name_velocity, name_acceleration);
    std::cout << R_0 << std::endl;
    int step = 0;
    /*data_files (name, coordinates, t);
    do {
        data_files (name, coordinates, t);
        Verlet_integration(coordinates, velocities);
        //std::cout << std::get<0>(velocities[0]) << std::endl;
        E = kinetic_energy_of_system(velocities);
        std::cout << E - E_init << '\t' << t << std::endl;
        t += dt;
        ++step;
    } while (is_equal(E, E_init) && t < simulation_time);
    plot(name, left_border, right_border, N, step);
*/
    std::string test = exec("rm -rf velocities && mkdir velocities");
    std::cout << Volume << std::endl;
    real_time_plotting(coordinates, velocities, name, left_border, right_border, N, E_init);
    return 0;
}


template<typename T, size_t... Is>
double distance_impl (T const& t, T const& t1, std::index_sequence<Is...>, std::index_sequence<Is...>) {
    return (std::sqrt((std::pow(std::get<Is>(t) - std::get<Is>(t1), 2) + ...)));
}

template <class Tuple>
double distance (const Tuple& t, const Tuple& t1) {
    constexpr auto size = std::tuple_size<Tuple>{};
    return distance_impl(t, t1, std::make_index_sequence<size>{}, std::make_index_sequence<size>{});
}


template<typename T, size_t... Is>
double vector_length_impl (T const& t, std::index_sequence<Is...>) {
    return std::sqrt((std::pow(std::get<Is>(t), 2) + ...));
}

template <class Tuple>
double vector_length (const Tuple& t) {
    constexpr auto size = std::tuple_size<Tuple>{};
    return vector_length_impl(t, std::make_index_sequence<size>{});
}


template<typename T, size_t... Is>
double scalar_product_impl (T const& t1, std::index_sequence<Is...>, T const& t2, std::index_sequence<Is...>) {
    return ((std::get<Is>(t1)*std::get<Is>(t2)) + ...);
}

template <class Tuple>
double scalar_product (const Tuple& t1, const Tuple& t2) {
    constexpr auto size = std::tuple_size<Tuple>{};
    return scalar_product_impl(t1, std::make_index_sequence<size>{}, t2, std::make_index_sequence<size>{});
}


// Returns the cosine of angle between two vectors (a, b).
double cos_ab (coord& a, coord& b) {
    return scalar_product(a, b) / (vector_length(a) * vector_length(b));
}


// result is a vector whose begin is point a and end is point b.
template<size_t Is = 0, typename... Tp>
void vector_creation (std::tuple<Tp...>& a, std::tuple<Tp...>& b, std::tuple<Tp...>& result) {
    std::get<Is>(result) = std::get<Is>(b) - std::get<Is>(a);
    if constexpr(Is + 1 != sizeof...(Tp))
        vector_creation<Is + 1>(a, b, result);
}

// Returns true if two tuples (t, t1) contains the same numbers.
template<typename T, size_t... Is>
bool equal_tuples_impl (T const& t, T const& t1, std::index_sequence<Is...>, std::index_sequence<Is...>) {
    return ((is_equal(std::get<Is>(t), std::get<Is>(t1))) & ...);
}

template <class Tuple>
bool equal_tuples (const Tuple& t, const Tuple& t1) {
    constexpr auto size = std::tuple_size<Tuple>{};
    return equal_tuples_impl(t, t1, std::make_index_sequence<size>{}, std::make_index_sequence<size>{});
}


// Offsets the vector to the frame of reference -- result vector = result.
template<size_t Is = 0, typename... Tp>
void vector_offset (std::tuple<Tp...>& vector, std::tuple<Tp...>& frame_of_reference, std::tuple<Tp...>& result) {
    std::get<Is>(result) = std::get<Is>(vector) + std::get<Is>(frame_of_reference);
    if constexpr(Is + 1 != sizeof...(Tp))
        vector_offset<Is + 1>(vector, frame_of_reference, result);
}


// std::to_string not safe enough. So it's realization via streams. It will be used everywhere instead of std::to_string.
template <typename T>
std::string toString (T val) {
    std::ostringstream oss;
    oss << val;
    return oss.str();
}


// Returns string contains tuple content.
template<typename T, size_t... Is>
std::string tuple_to_string_impl (T const& t, std::index_sequence<Is...>) {
    return ((toString(std::get<Is>(t)) + '\t') + ...);
}

template <class Tuple>
std::string tuple_to_string (const Tuple& t) {
    constexpr auto size = std::tuple_size<Tuple>{};
    return tuple_to_string_impl(t, std::make_index_sequence<size>{});
}


// Returns projection of point (q) to plane recorded in the form (A, B, C, D).
template<size_t Is = 0, typename... Tp>
void plane_projection (std::tuple<Tp...>& plane, coord& q) {
    if (!is_equal(std::get<Is>(plane), 0))
        std::get<Is>(q) *= (-1);
    if constexpr(Is + 2 != sizeof...(Tp))
        plane_projection<Is + 1>(plane, q);
}


// Returns the point of intersection the line (q1 q2) and the plane Ax+By+Cz+D = 0. r = q2 - q1.
coord intersection_point (coord& q1, coord& q2, double& A, double& B, double& C, double& D, coord& r) {

    double x0 = std::get<0>(q1);
    double y0 = std::get<1>(q1);
    double z0 = std::get<2>(q1);


    double p1 = std::get<0>(r);
    double p2 = std::get<1>(r);
    double p3 = std::get<2>(r);

    double t = - (D + A*x0 + B*y0 + C*z0) / (A*p1 + B*p2 + C*p3);

    double x = p1*t + x0;
    double y = p2*t + y0;
    double z = p3*t + z0;

    return std::make_tuple(x, y, z);
}


// The same as a last one but input only line (q1, q2) and plane recorde in form (A, B, C, D).
coord intersection_points_for_planes (coord& q1, coord& q2, planes& plane) {
    coord r;
    vector_creation(q1, q2, r);
    double A = std::get<0>(plane);
    double B = std::get<1>(plane);
    double C = std::get<2>(plane);
    double D = std::get<3>(plane);
    return std::move(intersection_point(q1, q2, A, B, C, D, r));
}


void next_step_position (coord& q1, coord& q2) {
    auto intersected_plane = area_borders[0];
    coord q, dq, closest_intersection, normal, moving, full_moving;
    double r, d = 1.0e308;
    for (auto & area_border : area_borders) {
        double A = std::get<0>(area_border);
        double B = std::get<1>(area_border);
        double C = std::get<2>(area_border);
        double D = std::get<3>(area_border);
        vector_creation(q1, q2, q);
        coord intersection = std::move(intersection_point(q1, q2, A, B, C, D, q));
        vector_creation(q1, intersection, dq);
        r = vector_length(dq);
        double error = 1.0e-10;
        if (std::fabs(cos_ab(dq, q) - 1) < error && r < d) {
            closest_intersection = intersection;
            moving = dq;
            full_moving = q;
            d = r;
            normal = std::make_tuple(A, B, C);
            intersected_plane = area_border;
        }
    }

    auto opposite_plane = intersected_plane;
    for (auto & area_border : area_borders) {
        double A = std::get<0>(area_border);
        double B = std::get<1>(area_border);
        double C = std::get<2>(area_border);
        coord test_normal = std::make_tuple(A, B, C);
        if (equal_tuples(normal, test_normal) && !equal_tuples(area_border, intersected_plane)) {
            opposite_plane = area_border;
            break;
        }
    }
    plane_projection(opposite_plane, closest_intersection);
    q2 = closest_intersection;
}


// Returns the std::vector of coordinates of main area's images.
std::vector<coord> areas_centers (double a) {
    std::vector<std::tuple<double, double, double>> centers;
    for (int k = 0; k < 3; ++k)
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i)
                centers.emplace_back(std::make_tuple(-a + i*a, -a + j*a, -a + k*a));
    return centers;
}


void computing (striiing &name, std::vector<coord>& coordinates, std::vector<coord>& velocities,
                double& E, double& t) {
    std::string name_q = std::get<0>(name);
    std::string name_v = std::get<1>(name);
    std::string name_a = std::get<2>(name);
    double t_debug = 1.0;
    bool flag = false;
    data_files_debug(name_q, coordinates, t_debug, flag);
    flag = false;
    data_files_debug(name_v, velocities, t_debug, flag);


    //s//td::string path = std::move(exec("cd velocities && echo $PWD"));
    //std::string V_names = path + "/velocities";
    //data_files(V_names, velocities, t);

    Verlet_integration_debug(coordinates, velocities, name_a);

    //Verlet_integration(coordinates, velocities);
    E = kinetic_energy_of_system(velocities);
    t += dt;
}


void real_time_plotting (std::vector<coord>& coordinates, std::vector<coord>& velocities,
                         striiing name, double min, double max, int number_of_points, double& E_init) {
    std::string range = "[" + toString(min) + ":" + toString(max) + "]";
    FILE *gp = popen("gnuplot  -persist", "w");
    if (!gp) throw std::runtime_error("Error opening pipe to GNUplot.");
    std::vector<std::string> stuff = {"set term pop",
                                      "set grid xtics ytics ztics",
                                      "set xrange " + range,
                                      "set yrange " + range,
                                      "set zrange " + range,
                                      "set key off",
                                      "set ticslevel 0",
                                      "set border 4095",
                                      "splot '-' u 1:2:3"};
    for (const auto& it : stuff)
        fprintf(gp, "%s\n", it.c_str());
    double E;
    double t; // !
    do {
        std::cout << t << std::endl;
        for (auto & coordinate : coordinates) {
            fprintf(gp, "%s\n%s\n", tuple_to_string(coordinate).c_str(), ",");
            //std::cout << '\n' << std::get<0>(coordinate) << "\n\n\n\n\n\n";
        }
        fprintf(gp, "%c\n%s\n", 'e', "splot '-' u 1:2:3");
        computing(name, coordinates, velocities, E, t);
        //std::cout << fabs(E-E_init) << '\t' << t << std::endl;
    } while (/*is_equal(E, E_init) && */t < simulation_time);
    fprintf(gp, "%c\n", 'q');
    pclose(gp);
}


// Creates a gif-file with molecular motion animation.
void plot (std::string name, double min, double max, int number_of_points, int& steps) {
    std::string range = "[" + toString(min) + ":" + toString(max) + "]";
    FILE *gp = popen("gnuplot  -persist", "w");
    if (!gp) throw std::runtime_error("Error opening pipe to GNUplot.");
    std::vector<std::string> stuff = {"set term gif animate delay 100",
                                      "set output \'" + name + ".gif\'",
                                      "set multiplot",
                                      "set grid xtics ytics ztics",
                                      "set xrange " + range,
                                      "set yrange " + range,
                                      "set zrange " + range,
                                      "set key off",
                                      "set ticslevel 0",
                                      "set border 4095",
                                      "do for [i = 0:" + toString(number_of_points-1) + "] {" ,
                                      "do for [j = 0:" + toString(steps-1) + "] {",
                                      "splot \'" + name + ".\'.i using 1:2:3 index j w linespoints pt 7,/",
                                      "} \ }",
                                      "q"};
    for (const auto& it : stuff)
        fprintf(gp, "%s\n", it.c_str());
    pclose(gp);
}


//The function returns the terminal ans. Input - string for term.
std::string exec (std::string str) {
    const char* cmd = str.c_str();
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
        result += buffer.data();
    result = result.substr(0, result.length()-1);
    return result;
}

// Clears the file.
void clear_data (std::string& file_name) {
    std::ofstream fout;
    fout.open(file_name, std::ofstream::out | std::ofstream::trunc);
    fout.close();
}


void data_file (std::string data_type, std::vector<coord>& data, double& t) {
    std::string path = std::move(exec("mkdir accelerations && cd accelerations && echo $PWD || cd accelerations/ && echo $PWD"));
    data_type = path + "/" + data_type;
    std::ofstream fout;
    data_type += ".txt";
    fout.open(data_type, std::ios::app);
    for (auto & i : data)
        fout << tuple_to_string(i) << t << std::endl;
    fout << std::endl;
    fout.close();
}


void data_files (std::string& name, std::vector<coord>& data, double& t) {
    static bool flag = false;
    for (int i = 0; i < data.size(); ++i) {
        std::ofstream fout;
        fout.open(name + '.' + toString(i), std::ios::app);
        std::string buf = std::move(tuple_to_string(data[i]));
        fout << buf << t << ((flag) ? "\n\n\n\n" + buf + "\n" : "\n");
        fout.close();
    }
    flag = true;
}

void data_files_debug (std::string& name, std::vector<coord>& data, double& t_debug, bool& flag) {
    //static bool flag = false;
    for (int i = 0; i < data.size(); ++i) {
        std::ofstream fout;
        fout.open(name + '.' + toString(i), std::ios::app);
        std::string buf = std::move(tuple_to_string(data[i]));
        fout << buf << t_debug << ((flag) ? "\n\n\n\n" + buf + "\n" : "\n");
        fout.close();
    }
    flag = true;
}


// Used for circuit stability check.
double kinetic_energy_of_system (std::vector<coord>& velocities) {
    double E = 0;
    for (int i = 0; i < N; ++i) {
        double v2 = scalar_product(velocities[i], velocities[i]);
        if (std::isfinite(v2))
            E += m * v2 / 2.0;
    }
    return E;
}


// We can't compare two doubles without an error. So it returns true if the distance between two doubles less than
// standard error.
bool is_equal (double a, double b) {
    return std::fabs(a - b) < std::numeric_limits<double>::epsilon();
}


// Function returns true, if std::vectors of tuples are same. Estimates with error of computer representation of doubles.
bool is_same (std::vector<coord>& a_1, std::vector<coord>& a_2) {
    bool ans = true;
    int i = 0;
    do {
        ans &= equal_tuples(a_1[i], a_2[i]);
        ++i;
    } while (ans && i < a_1.size());
    return ans;
}

// Template returns true if particle outside the main area.
template<typename T, size_t... Is>
bool border_intersection_impl (T const& t, std::index_sequence<Is...>) {
    return ((std::fabs(std::get<Is>(t)) > right_border) | ...);
}

template <class Tuple>
bool border_intersection (const Tuple& t) {
    constexpr auto size = std::tuple_size<Tuple>{};
    return border_intersection_impl(t, std::make_index_sequence<size>{});
}


void periodic_borders (std::vector<coord>& q1, std::vector<coord>& q2) {
    for (int i = 0; i < q2.size(); ++i)
        if (border_intersection(q2[i]))
            next_step_position(q1[i], q2[i]);
}


// Template contains the equation of movement for particles.
template<size_t Is = 0, typename... Tp>
void coordinates_equations (std::tuple<Tp...>& q, std::tuple<Tp...>& v, std::tuple<Tp...>& a) {
    std::get<Is>(q) += std::get<Is>(v)*dt + std::get<Is>(a)*std::pow(dt, 2) / 2.0;
    if constexpr(Is + 1 != sizeof...(Tp))
        coordinates_equations<Is + 1>(q, v, a);
}


// Template contains the equation for velocities of particles.
template<size_t Is = 0, typename... Tp>
void velocities_equations (std::tuple<Tp...>& v, std::tuple<Tp...>& a_current, std::tuple<Tp...>& a_next) {
    std::get<Is>(v) += (std::get<Is>(a_current) + std::get<Is>(a_next)) / 2.0 * dt;
    if constexpr(Is + 1 != sizeof...(Tp))
        velocities_equations<Is + 1>(v, a_current, a_next);
}

// Input: vectors coordinates and velocities. Output: vector of coordinates after one time-step.
void Verlet_integration (std::vector<coord>& q, std::vector<coord>& v) {
    static bool flag = false;
    std::vector<coord> a_next, a_current, q_previous;
    std::vector<coord> a (N);
    if (flag) q_previous = q;
    for (int i = 0; i < N; ++i) {
        a_current = std::move(total_particle_acceleration(q));
        coordinates_equations(q[i], v[i], a_current[i]);
        a_next = std::move(total_particle_acceleration(q));
        vector_creation(a_current[i], a_next[i], a[i]);
//        std::string name = "a";
//        name = PATH + name;
        double t = dt;
//        data_files(name, a, t);
      velocities_equations(v[i], a_current[i], a_next[i]);
    }
    if (flag) periodic_borders(q_previous, q);
    flag = true;
    //for (auto & i : q) periodic_borders(i);
    momentum_exchange(q, v);
}

void Verlet_integration_debug (std::vector<coord>& q, std::vector<coord>& v, std::string& acceleration_files_name) {
    static std::vector<coord> a_current = std::move(total_particle_acceleration(q));
    static std::vector<coord> q_current;
    double debug_t = 0;
    bool flag = false;
    data_files_debug(acceleration_files_name, a_current, debug_t, flag);
    static bool first_step = true;
    std::vector<coord> a_next;
    for (int i = 0; i < N; ++i)
        coordinates_equations(q[i], v[i], a_current[i]);
    if (!first_step) {
        a_next = std::move(total_particle_acceleration(q));
        periodic_borders(q_current, q);
        for (int i = 0; i < N; ++i)
            velocities_equations(v[i], a_current[i], a_next[i]);
        a_current = a_next;
    }
    first_step = false;
    q_current = q;
}





// If two particles impact they exchange momentums of each other.
// Input: vector coordinates, vector velocities. Last changes by reference.
void momentum_exchange (std::vector<coord>& coordinates, std::vector<coord>& velocities) {
    //static int colisions = 0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (distance(coordinates[i], coordinates[j]) <= 2.0*R_Ar && i != j) {
                coord buf = velocities[i];
                velocities[i] = velocities[j];
                velocities[j] = buf;
                //++colisions;
            }
    //std::cout << colisions << std::endl;
}


// The function defines the closest "second" particle from the main and image-area.
coord minimal_distance (coord& q1, coord& q2, double& R_ij) {
    double test;
    R_ij = 1.0e300;
    coord offseted, result;
    for (auto & neighboring_cube : neighboring_cubes) {
        vector_offset(q2, neighboring_cube, offseted);
        test = distance(q1, offseted);
        if (test < R_ij) {
            R_ij = test;
            result = offseted;
        }
    }
    return result;
}


// Returns the projections of acceleration of particle q1 to coordinate axes.
template<size_t Is = 0, typename... Tp>
void acceleration_projections (std::tuple<Tp...>& a, std::tuple<Tp...>& q1, std::tuple<Tp...>& q2, double& distance) {
    double F = (distance <= R_max) ? single_force(std::get<Is>(q2)) / 2.0 : 0;
    std::get<Is>(a) += (std::isfinite(F) ? F/m : 0);
    if constexpr(Is + 1 != sizeof...(Tp))
        acceleration_projections<Is + 1>(a, q1, q2, distance);
}


/* Returns vector of accelerations for particles. It's the most time-consuming operation, so it computing in parallels
 * via omp.h. I don't know what more effective: using it in parallel or just using -O3 flag.
 * Input: vector of coordinates. */
std::vector<coord> total_particle_acceleration (std::vector<coord>& particles) {
    std::vector<coord> acceleration;
#pragma omp parallel
    {
        std::vector<coord> acceleration_private;
        double a_x, a_y, a_z, R_ij;
#pragma omp for nowait schedule(static)
        for (int i = 0; i < N; ++i) {
            a_x = a_y = a_z = 0;
            coord a = std::make_tuple(0, 0, 0);
            for (int j = 0; j < N; ++j)
                if (i != j) {
                    R_ij = 0; // Distance between two interaction particles. ATTENTION: takes on values in
                              // coord minimal_distance (coord& q1, coord& q2, double& R_ij) via reference!
                    coord second_particle = std::move(minimal_distance(particles[i], particles[j], R_ij));
                    acceleration_projections(a, particles[i], second_particle, R_ij);
                    //a_x += single_force(std::get<0>(particles[i]) - std::get<0>(particles[j])) / m;
                    //a_y += single_force(std::get<1>(particles[i]) - std::get<1>(particles[j])) / m;
                    //a_z += single_force(std::get<2>(particles[i]) - std::get<2>(particles[j])) / m;
                }
            acceleration_private.emplace_back(a);
            //acceleration_private.emplace_back(std::make_tuple(a_x, a_y, a_z));
        }

#pragma omp for schedule(static) ordered
        for (int i = 0; i < omp_get_num_threads(); ++i) {
#pragma omp ordered
            acceleration.insert(acceleration.end(), acceleration_private.begin(), acceleration_private.end());
        }
    }
    return acceleration;
}


// Returns force of two-particles interaction via Lennard-Jones potential. Input distance between two particles.
double single_force (double& R) {
    return 24.0 * Theta / R * (2.0 * pow(sigma / R, 12.0) - pow(sigma / R, 6.0));
}

// Returns the projection of initial velocity to coordinate axes.
coord velocity_direction (double& cos_psi, double& mu, double& cos_gamma) {
    return std::make_tuple(cos_psi*V_init, mu*V_init, cos_gamma*V_init);
}


#include <iostream>

std::vector<coord> initial_velocities_debug () {
    double epsilon = 1.0 / N;
    std::vector<coord> velocities;
    coord direction;
    //std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < N; ++i) {
        double mu, a, b, cos_psi, cos_gamma, d = 10;
        do {
            mu = 2 * epsilon * (rand()%(N+1)) - 1;
            do {
                a = 2 * epsilon * (rand()%(N+1)) - 1;
                b = 2 * epsilon * (rand()%(N+1)) - 1;
                d = std::pow(a, 2) + std::pow(b, 2);
            } while (d > 1);
            cos_psi = a / std::sqrt(d);
            cos_gamma = std::sqrt(1.0 - (std::pow(mu, 2) + std::pow(cos_psi, 2))) *
                    ((i%2 == 0) ? 1.0 : (-1.0));
        } while (std::pow(mu, 2) + std::pow(cos_psi, 2) > 1);
        velocities.emplace_back(std::move(velocity_direction(cos_psi, mu, cos_gamma)));
    }
    return velocities;
}


// Returns initial coordinates of particles evenly distributed over the volume.
std::vector<coord> initial_coordinates_debug () {
    std::cout << std::endl << right_border << std::endl;
    std::vector<coord> coordinates;
    //std::uniform_real_distribution<> dis(left_border, right_border);
    //int width = characteristic_size
    for (int i = 0; i < N; ++i) {
        double x = left_border + 1.0e-11 * (rand() % 80298*2);
        double y = left_border + 1.0e-11 * (rand() % 80298*2);
        double z = left_border + 1.0e-11 * (rand() % 80298*2);
        coordinates.emplace_back(std::move(std::make_tuple(x, y, z)));
    }
    return coordinates;
}



// Returns vector of velocities for every particle.
/*std::vector<coord> initial_velocities () {
    std::vector<coord> velocities;
    coord direction;
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < N; ++i) {
        double mu, a, b, cos_psi, cos_gamma, d = 10;
        do {
            mu = 2 * dis(gen) - 1;
            do {
                a = 2 * dis(gen) - 1;
                b = 2 * dis(gen) - 1;
                d = std::pow(a, 2) + std::pow(b, 2);
            } while (d > 1);
            cos_psi = a / std::sqrt(d);
            cos_gamma = std::sqrt(1.0 - (std::pow(mu, 2) + std::pow(cos_psi, 2))) *
                    ((dis(gen) > 0.5) ? 1.0 : (-1.0));
        } while (std::pow(mu, 2) + std::pow(cos_psi, 2) > 1);
        velocities.emplace_back(std::move(velocity_direction(cos_psi, mu, cos_gamma)));
    }
    return velocities;
}

// Returns initial coordinates of particles evenly distributed over the volume.
std::vector<coord> initial_coordinates () {
    std::vector<coord> coordinates;
    std::uniform_real_distribution<> dis(left_border, right_border);
    for (int i = 0; i < N; ++i) {
        double x = dis(gen);
        double y = dis(gen);
        double z = dis(gen);
        coordinates.emplace_back(std::move(std::make_tuple(x, y, z)));
    }
    return coordinates;
}*/