/* Program of 3d-modeling gas behavior in unit volume via molecular dynamics method.
 * Interaction of particles modeling via Lennard-Jones potential.
 * Coordinates and velocities defines via Verlet integration.
 * Initial coordinates and velocities directions distributed evenly.
 * By vector in comments it should be understood std::vector<std::tuple<>>,
 * by coordinates, velocities and accelerations it should be understood std::tuple contains projections on coordinate
 * axes. */

#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <utility>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <array>
#include <algorithm>
#include <iterator>
#include <memory>
#include <stdexcept>
#include "omp.h"


// Program constants:
const int N = 1e3; //Number of particles
const double dt = 1.0e-10; // Time-step
const double left_border = -0.5;
const double right_border = 0.5;
bool realtime = false;


// Fundamental constants
const double k = 1.380649e-16; // Boltzmann constant, erg/K

// Substance characteristics
const std::string substance = "Ar";
const double m = 6.6335e-23;  // Argon mass, g
const double eps = 119.8;  // Potential pit depth (Theta/k), K
const double Theta = k*eps;  // Equilibrium temperature
const double sigma = 3.405e-8;  // Smth like shielding length, cm
const double R_0 = sigma * pow(2, 1.0/6.0);

// Environment
const double T = 300; // Temperature, K
const double V_init = std::sqrt(3*k*T / m);  // Initial particles velocity

typedef std::tuple<double, double, double> coord;

std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()


std::vector<coord> default_coordinates ();

std::vector<coord> default_velocities ();

double single_force (double R);

std::vector<coord> total_particle_acceleration (std::vector<coord>& particles);

void momentum_exchange (std::vector<coord>& coordinates, std::vector<coord>& velocities);

void Verlet_integration (std::vector<coord>& q, std::vector<coord>& v);

void velocities_equations (coord& v, coord& a_current, coord& a_next);

void coordinate_equations (coord& q, coord& v, coord& a);

bool is_equal (double a, double b);

bool is_same (std::vector<coord>& a_1, std::vector<coord>& a_2);

double energy_of_system (std::vector<coord>& velocities);

void data_file (std::string data_type, std::vector<coord>& data, double& t);

void data_files (std::string& name, std::vector<coord>& data, double& t);

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

int main () {
    std::vector<coord> coordinates = std::move(default_coordinates());
    std::vector<coord> velocities = std::move(default_velocities());
    double E, E_init = N * m*std::pow(V_init, 2)/2.0;
    double t = 0;
    std::string name = "Ar_coordinates";
    if (!realtime) {
        std::string path = std::move(exec("rm -rf trajectories && mkdir trajectories && cd trajectories && echo $PWD"));
        name = path + '/' + name;
    }
    int step = 0;
    data_files (name, coordinates, t);
    do {
        Verlet_integration(coordinates, velocities);
        //std::cout << std::get<0>(velocities[0]) << std::endl;
        E = energy_of_system(velocities);
        std::cout << E - E_init << '\t' << t << std::endl;
        t += dt;
        data_files (name, coordinates, t);
        ++step;
    } while (is_equal(E, E_init) && t < 1.0e3*dt);
    plot(name, left_border, right_border, N, step);
    return 0;
}



void plot (std::string name, double min, double max, int number_of_points, int& steps) {
    std::string range = "[" + std::to_string(min) + ":" + std::to_string(max) + "]";
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
                                      "do for [i = 0:" + std::to_string(number_of_points-1) + "] {" ,
                                            "do for [j = 0:" + std::to_string(steps-1) + "] {",
                                                "splot \'" + name + ".\'.i using 1:2:3 index j w linespoints pt 7,/",
                                      "} \ }",
                                      "q"};
    for (const auto& it : stuff)
        fprintf(gp, "%s\n", it.c_str());
    pclose(gp);
}


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


void clear_data (std::string& file_name) {
    std::ofstream fout;
    fout.open(file_name, std::ofstream::out | std::ofstream::trunc);
    fout.close();
}

// REFACTOR OTHERS LIKE THIS!
template<typename T, size_t... Is>
std::string tuple_to_string_impl(T const& t, std::index_sequence<Is...>) {
    return (((std::to_string(std::get<Is>(t)) + '\t') + ...));
}

template <class coord>
std::string tuple_to_string (const coord& t) {
    constexpr auto size = std::tuple_size<coord>{};
    return tuple_to_string_impl(t, std::make_index_sequence<size>{});
}

void data_file (std::string data_type, std::vector<coord>& data, double& t) {
    std::ofstream fout;
    data_type += ".txt";
    fout.open(data_type, std::ios::app);
    for (auto & i : data)
        fout << tuple_to_string(i) << t << std::endl;
    fout << std::endl;
    fout.close();
}


void data_files (std::string& name, std::vector<coord>& data, double& t) {
    for (int i = 0; i < data.size(); ++i) {
        std::ofstream fout;
        fout.open(name + '.' + std::to_string(i), std::ios::app);
        fout << tuple_to_string(data[i]) << t << std::endl << std::endl;
        fout.close();
    }
}

template<typename T, size_t... Is>
auto scalar_square_impl(T const& t, std::index_sequence<Is...>) {
    return ((std::pow(std::get<Is>(t), 2) + ...));
}

template <class Tuple>
double scalar_square (const Tuple& t) {
    constexpr auto size = std::tuple_size<Tuple>{};
    return scalar_square_impl(t, std::make_index_sequence<size>{});
}

// Used for circuit stability check.
double energy_of_system (std::vector<coord>& velocities) {
    double E = 0;
    for (int i = 0; i < N; ++i)
        E += m*scalar_square(velocities[i]) / 2.0;
    return E;
}


bool is_equal (double a, double b) {
    return std::fabs(a - b) < std::numeric_limits<double>::epsilon();
}

template<typename T, size_t... Is>
auto equal_impl (T const& t, T const& t1, std::index_sequence<Is...>, std::index_sequence<Is...>) {
    return ((is_equal(std::get<Is>(t), std::get<Is>(t1))) & ...);
}

template <class Tuple>
bool equal_tuples (const Tuple& t, const Tuple& t1) {
    constexpr auto size = std::tuple_size<Tuple>{};
    return equal_impl(t, t1, std::make_index_sequence<size>{}, std::make_index_sequence<size>{});
}

// Function returns true, if vectors of tuples are same. Estimates with error of computer representation of doubles.
bool is_same (std::vector<coord>& a_1, std::vector<coord>& a_2) {
    bool ans = true;
    int i = 0;
    do {
        ans &= equal_tuples(a_1[i], a_2[i]);
        ++i;
    } while (ans && i < N);
    return ans;
}

template<size_t Is = 0, typename... Tp>
void periodic_borders (std::tuple<Tp...>& t) {
    if (std::fabs(std::get<Is>(t)) >= right_border)
        std::get<Is>(t) = -std::get<Is>(t);
    if constexpr(Is + 1 != sizeof...(Tp))
        periodic_borders<Is + 1>(t);
}

/* Well, using it this way is a bad way. Better to write templates, but it is necessary to change the architecture of
 * the program. So two functions bellow just difference equations of Verlet integration.
 * Input: vectors of coordinates (velocities) in tuple. Change by reference. */
void coordinate_equations (coord& q, coord& v, coord& a) {
    //debug_tuple_output(v);
    //std::cout << std::endl;
    std::get<0>(q) += std::get<0>(v)*dt + std::get<0>(a)*std::pow(dt, 2)/2.0;
    std::get<1>(q) += std::get<1>(v)*dt + std::get<1>(a)*std::pow(dt, 2)/2.0;
    std::get<2>(q) += std::get<2>(v)*dt + std::get<2>(a)*std::pow(dt, 2)/2.0;
}

void velocities_equations (coord& v, coord& a_current, coord& a_next) {
    std::get<0>(v) += (std::get<0>(a_next) + std::get<0>(a_current)) / 2.0 * dt;
    std::get<1>(v) += (std::get<1>(a_next) + std::get<1>(a_current)) / 2.0 * dt;
    std::get<2>(v) += (std::get<2>(a_next) + std::get<2>(a_current)) / 2.0 * dt;
}

// Input: vectors coordinates and velocities. Output: vector of coordinates after one time-step.
void Verlet_integration (std::vector<coord>& q, std::vector<coord>& v) {
    std::vector<coord> a_next, a_current;
    for (int i = 0; i < N; ++i) {
        a_current = total_particle_acceleration(q);
        double debug_t = i;
        data_file("accelerations"+std::to_string(i), a_current, debug_t);
        coordinate_equations(q[i], v[i], a_current[i]);
        a_next = total_particle_acceleration(q);
        //if (!is_same(a_current, a_next))
            velocities_equations(v[i], a_current[i], a_next[i]);
        periodic_borders(q[i]);
    }
    momentum_exchange(q, v);
}



template<typename T, size_t... Is>
auto distance_impl (T const& t, T const& t1, std::index_sequence<Is...>, std::index_sequence<Is...>) {
    return (std::sqrt((std::pow(std::get<Is>(t) - std::get<Is>(t1), 2) + ...)));
}

template <class Tuple>
double distance (const Tuple& t, const Tuple& t1) {
    constexpr auto size = std::tuple_size<Tuple>{};
    return distance_impl(t, t1, std::make_index_sequence<size>{}, std::make_index_sequence<size>{});
}

// If two particles impact they exchange momentums of each other.
// Input: vector coordinates, vector velocities. Last changes by reference.
void momentum_exchange (std::vector<coord>& coordinates, std::vector<coord>& velocities) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (distance(coordinates[i], coordinates[j]) <= R_0 && i != j) {
                coord buf = velocities[i];
                velocities[i] = velocities[j];
                velocities[j] = buf;
            }
}


/* Returns vector of accelerations for particles. It's the most time-consuming operation, so it computing in parallels
 * via omp.h. I don't know what more effective: using it in parallel or just using -O3 flag.
 * Input: vector of coordinates. */
/*std::vector<coord> total_particle_acceleration (std::vector<coord>& particles) {
    std::vector<coord> acceleration;
    double a_x, a_y, a_z;
    for (int i = 0; i < N; ++i) {
        a_x = a_y = a_z = 0;
        for (int j = 0; j < N; ++j)
            if (i != j && distance(particles[i], particles[j]) <= 3.0 * R_0) {
                a_x += single_force(std::get<0>(particles[i]) - std::get<0>(particles[j])) / m;
                a_y += single_force(std::get<1>(particles[i]) - std::get<1>(particles[j])) / m;
                a_z += single_force(std::get<2>(particles[i]) - std::get<2>(particles[j])) / m;
            }
        acceleration.emplace_back(std::make_tuple(a_x, a_y, a_z));
        }
    return acceleration;

}*/
std::vector<coord> total_particle_acceleration (std::vector<coord>& particles) {
    std::vector<coord> acceleration;
#pragma omp parallel
    {
        std::vector<coord> acceleration_private;
        double a_x, a_y, a_z;
#pragma omp for nowait schedule(static)
        for (int i = 0; i < N; ++i) {
            a_x = a_y = a_z = 0;
            for (int j = 0; j < N; ++j)
                if (i != j && distance(particles[i], particles[j]) <= 3.0 * R_0) {
                    a_x += single_force(std::get<0>(particles[i]) - std::get<0>(particles[j])) / m;
                    a_y += single_force(std::get<1>(particles[i]) - std::get<1>(particles[j])) / m;
                    a_z += single_force(std::get<2>(particles[i]) - std::get<2>(particles[j])) / m;
                }
            acceleration_private.emplace_back(std::make_tuple(a_x, a_y, a_z));
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
double single_force (double R) {
    return 24.0 * Theta / R * (2.0 * pow(sigma / R, 12.0) - pow(sigma / R, 6.0));
}


coord velocity_direction (double& cos_psi, double& mu, double& cos_gamma) {
    return std::make_tuple(cos_psi*V_init, mu*V_init, cos_gamma*V_init);
}

// Returns vector of velocities for every particles.
std::vector<coord> default_velocities () {
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
            cos_gamma = std::sqrt(1.0 - (std::pow(mu, 2) + std::pow(cos_psi, 2)));
        } while (std::pow(mu, 2) + std::pow(cos_psi, 2) > 1);
        velocities.push_back(std::move(velocity_direction(cos_psi, mu, cos_gamma)));
    }
    return velocities;
}

// Returns initial coordinates of particles evenly distributed over the volume.
std::vector<coord> default_coordinates () {
    std::vector<coord> coordinates;
    std::uniform_real_distribution<> dis(left_border, right_border);
    for (int i = 0; i < N; ++i) {
        double x = dis(gen);
        double y = dis(gen);
        double z = dis(gen);
        coordinates.emplace_back(std::move(std::make_tuple(x, y, z)));
    }
    return coordinates;
}