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
#include <memory>
#include <stdexcept>
#include <sstream>


// Fundamental constants
const double k_B = 1.380649e-16; // Boltzmann constant, erg/K


// Substance characteristics
const std::string substance = "Ar";
const double m = 6.6335e-23;  // Argon mass, g
const double eps = 119.8; // Potential pit depth (Theta/k_B), K
const double Theta = k_B * eps;
const double sigma = 3.405e-8; // Zero potential.
const double R_0 = sigma * std::pow(2.0, 1.0/6.0); // Minimum potential
const double R_Ar = 1.54e-8;


// Gas characteristics
const double P = 1.0e6;
const double T = 160.0; // Temperature, K
const double n = P / k_B / T; // Concentration, 1/cm^-3
const double V_init = std::sqrt(3.0 * k_B * T / m); // rms speed corresponding to a given temperature (T), cm/c


// Program constants
const int N = 1e2; //Number of particles
const double dt = 1.0e-10; // Time-step, c
const double simulation_time = 7e-8;
const double R_cutoff = 2.0 * R_0;


// Model constants
const double Volume = N/n; // n corresponds to a unit volume
const double characteristic_size = std::pow(Volume, 1.0/3.0);
const double left_border = -characteristic_size / 2.0;
const double right_border = characteristic_size / 2.0;


typedef std::tuple<double, double, double> coord; // Contains coordinates as (x, y, z)


std::vector<coord> neighboring_cubes; // Contains coordinates of centers virtual areas


std::vector<coord> areas_centers (double a); // Filling the vector upper


std::random_device rd;  // Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()


std::vector<coord> initial_coordinates ();

std::vector<coord> initial_velocities ();

std::string exec (const std::string &str);

void Verlet_integration (std::vector<coord>& q, std::vector<coord>& v);

bool is_equal (double a, double b);

void data_file (std::string& name, std::vector<coord>& data, int& step);

void frames (std::string& name, int& step);


int main () {
    // Just ancillary code for define directories with files,
    std::string trajectory_files_path = std::move(exec("rm -rf trajectories && mkdir trajectories && cd trajectories && echo $PWD"));
    std::string trajectory_files_name = trajectory_files_path + '/' + substance + "_coordinates";
    std::string velocities_files_path = std::move(exec("rm -rf velocities && mkdir velocities && cd velocities && echo $PWD"));
    std::string velocities_files_name = velocities_files_path + '/' + substance + "_velocities";


    // Initials.
    std::vector<coord> coordinates = std::move(initial_coordinates());
    std::vector<coord> velocities = std::move(initial_velocities());
    neighboring_cubes = std::move(areas_centers(characteristic_size));

    double t = 0;
    int step = 0;
    do {
        data_file(trajectory_files_name, coordinates, step);
        data_file(velocities_files_name, coordinates, step);
        Verlet_integration(coordinates, velocities);
        t += dt;
        ++step;
    } while (t < simulation_time);
    frames(trajectory_files_name, step);

    // exec ("cd " + trajectory_files_path + "&& convert *.jpg out.gif"); // Creates gif... very slow.

    return 0;
}


// We can't compare two doubles without an error. So it returns true if the distance between two doubles less than
// standard error.
bool is_equal (double a, double b) {
    return std::fabs(a - b) < std::numeric_limits<double>::epsilon();
}

template<typename T, size_t... Is>
bool equal_tuples_impl (T const& t, T const& t1, std::index_sequence<Is...>, std::index_sequence<Is...>) {
    return ((is_equal(std::get<Is>(t), std::get<Is>(t1))) & ...);
}

// Returns true if two tuples (t, t1) contains the same numbers.
template <class Tuple>
bool equal_tuples (const Tuple& t, const Tuple& t1) {
    constexpr auto size = std::tuple_size<Tuple>{};
    return equal_tuples_impl(t, t1, std::make_index_sequence<size>{}, std::make_index_sequence<size>{});
}


template<typename T, size_t... Is>
double distance_impl (T const& t, T const& t1, std::index_sequence<Is...>, std::index_sequence<Is...>) {
    return (std::sqrt((std::pow(std::get<Is>(t) - std::get<Is>(t1), 2) + ...)));
}

// Returns the distance between two points.
template <class Tuple>
double distance (const Tuple& t, const Tuple& t1) {
    constexpr auto size = std::tuple_size<Tuple>{};
    return distance_impl(t, t1, std::make_index_sequence<size>{}, std::make_index_sequence<size>{});
}


// result is a vector (coord) whose begin is point a and end is point b.
template<size_t Is = 0, typename... Tp>
void vector_creation (std::tuple<Tp...>& a, std::tuple<Tp...>& b, std::tuple<Tp...>& result) {
    std::get<Is>(result) = std::get<Is>(b) - std::get<Is>(a);
    if constexpr(Is + 1 != sizeof...(Tp))
        vector_creation<Is + 1>(a, b, result);
}


// Offsets the vector to the frame of reference -- result vector = result.
template<size_t Is = 0, typename... Tp>
void vector_offset (std::tuple<Tp...>& vector, std::tuple<Tp...>& frame_of_reference, std::tuple<Tp...>& result) {
    std::get<Is>(result) = std::get<Is>(vector) + std::get<Is>(frame_of_reference);
    if constexpr(Is + 1 != sizeof...(Tp))
        vector_offset<Is + 1>(vector, frame_of_reference, result);
}


template<size_t Is = 0, typename... Tp>
void vector_scalar_multiplication (std::tuple<Tp...>& vector, double lambda, std::tuple<Tp...>& result) {
    std::get<Is>(result) = std::get<Is>(vector) * lambda;
    if constexpr(Is + 1 != sizeof...(Tp))
        vector_scalar_multiplication<Is + 1>(vector, lambda, result);
}


// std::to_string not safe enough. It will be used everywhere instead of std::to_string.
template <typename T>
std::string toString (T val) {
    std::ostringstream oss;
    oss << val;
    return oss.str();
}

// Filling tuple (coordinate) with random doubles in range from left to right.
template<size_t Is = 0, typename... Tp>
void random_tuple (std::tuple<Tp...>& coordinate, double left, double right) {
    std::uniform_real_distribution<> dis(left, right);
    std::get<Is>(coordinate) = dis(gen);
    if constexpr(Is + 1 != sizeof...(Tp))
        random_tuple<Is + 1>(coordinate, left, right);
}


bool good_distance (coord& particle, std::vector<coord>& particles) {
    bool ans = true;
    for (auto & i : particles)
        ans &= (distance(particle, i) >= R_cutoff);
    return ans;
}


// Returns initial coordinates of particles evenly distributed over the volume.
std::vector<coord> initial_coordinates () {
    std::vector<coord> coordinates;
    coord coordinate;
    coordinates.emplace_back(std::make_tuple(0, 0, 0));
    for (int i = 1; i < N; ++i) {
        do {
            random_tuple(coordinate, left_border, right_border);
        } while (!good_distance(coordinate, coordinates));
        coordinates.emplace_back(coordinate);
    }
    return coordinates;
}


// Returns uniform distributed in direction velocities with same absolute values.
std::vector<coord> initial_velocities () {
    std::vector<coord> velocities;
    coord direction;
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < N; ++i) {
        double mu, a, b, cos_psi, cos_gamma, d = 10;
        do {
            mu = 2 * dis(gen) - 1.0;
            do {
                a = 2 * dis(gen) - 1.0;
                b = 2 * dis(gen) - 1.0;
                d = std::pow(a, 2.0) + std::pow(b, 2.0);
            } while (d > 1);
            cos_psi = a / std::sqrt(d);
            cos_gamma = std::sqrt(1.0 - (std::pow(mu, 2.0) + std::pow(cos_psi, 2.0))) *
                        ((dis(gen) > 0.5) ? 1.0 : (-1.0));
        } while (std::pow(mu, 2) + std::pow(cos_psi, 2) >= 1.0);
        coord velocity_direction = std::make_tuple(cos_psi, mu, cos_gamma);
        vector_scalar_multiplication(velocity_direction, V_init, velocity_direction);
        velocities.emplace_back(std::move(velocity_direction));
    }
    return velocities;
}


// Returns the coordinates of main area's images.
std::vector<coord> areas_centers (double a) {
    std::vector<std::tuple<double, double, double>> centers;
    for (int k = 0; k < 3; ++k)
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i)
                centers.emplace_back(std::make_tuple(-a + i*a, -a + j*a, -a + k*a));
    return centers;
}


// Defines the closest particle image in all related regions.
coord closest_particle_image (coord& q1, coord& q2, double& R_ij) {
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


// Force of two-particles interaction via LJ-potential.
double single_force (double R) {
    return 24.0 * Theta / R * (2.0 * std::pow(sigma / R, 12.0) - std::pow(sigma / R, 6.0));
}


template<size_t Is = 0, typename... Tp>
void acceleration_projections (std::tuple<Tp...>& a, std::tuple<Tp...>& q1, std::tuple<Tp...>& q2, double& R_ij) {
    double F = (R_ij <= R_cutoff && R_ij > 2.0*R_Ar) ? single_force(std::fabs(std::get<Is>(q1) - std::get<Is>(q2))) : 0;
    std::get<Is>(a) += (std::isfinite(F) ? F/m : 0);
    if constexpr(Is + 1 != sizeof...(Tp))
        acceleration_projections<Is + 1>(a, q1, q2, R_ij);
}


// Will be rewrited with OMP.
std::vector<coord> total_particle_acceleration (std::vector<coord>& particles) {
    std::vector<coord> a (N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            if (i != j) {
                double R_ij;
                coord second_particle = std::move(closest_particle_image(particles[i], particles[j], R_ij));
                acceleration_projections(a[i], particles[i], second_particle, R_ij);
            }
    }
    return a;
}


double border_intersection_count (double& coordinate) {
    double count;
    modf(std::fabs(coordinate/right_border), &count);
    return count;
}


template<size_t Is = 0, typename... Tp>
void periodic_boundary_conditions (std::tuple<Tp...>& q) {
    double intersected_borders = border_intersection_count(std::get<Is>(q));
    if (std::get<Is>(q) < left_border) std::get<Is>(q) += intersected_borders * right_border;
    if (std::get<Is>(q) >= right_border) std::get<Is>(q) += intersected_borders * left_border;
    if constexpr(Is + 1 != sizeof...(Tp))
        periodic_boundary_conditions<Is + 1>(q);
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


void Verlet_integration (std::vector<coord>& q, std::vector<coord>& v) {
    static bool first_step = true;
    static std::vector<coord> a_current;

    if (first_step) a_current = std::move(total_particle_acceleration(q));

    std::vector<coord> q_current (N);
    std::vector<int> outsiders;
    std::vector<coord> a_next;

    // Definition of coordination on next time step:
    for (int i = 0; i < N; ++i) {
        coordinates_equations(q[i], v[i], a_current[i]);
        q_current[i] = q[i];
        periodic_boundary_conditions(q[i]);
        if (!equal_tuples(q_current[i], q[i])) outsiders.emplace_back(i);
    }

    // Definition next time step velocities:
    if (!first_step) {
        a_next = std::move(total_particle_acceleration(q_current));
        for (int i = 0; i < N; ++i)
            for (int & outsider : outsiders) {
                if (i == outsider)
                    ++i;
                else
                    velocities_equations(v[i], a_current[i], a_next[i]);
            }
        a_current = a_next;
    }
    first_step = false;
}


//The function returns the terminal ans. Input - string for term.
std::string exec (const std::string& str) {
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


void data_file (std::string& name, std::vector<coord>& data, int& step) {
    std::ofstream fout;
    fout.open(name + '.' + toString(step), std::ios::app);
    for (auto & i : data)
        fout << tuple_to_string(i) << std::endl;
    fout.close();
}


void frames (std::string& name, int& step) {
    std::string range = "[" + toString(left_border) + ":" + toString(right_border) + "]";
    for (int i = 0; i < step-1; ++i) {
        FILE *gp = popen("gnuplot  -persist", "w");
        if (!gp) throw std::runtime_error("Error opening pipe to GNUplot.");
        std::vector<std::string> stuff = {"set term jpeg size 700, 700",
                                          "set output \'" + name + toString(i) + ".jpg\'",
                                          "set grid xtics ytics ztics",
                                          "set xrange " + range,
                                          "set yrange " + range,
                                          "set zrange " + range,
                                          "set key off",
                                          "set ticslevel 0",
                                          "set border 4095",
                                          "splot \'" + name + "." + toString(i) + "\' using 1:2:3 pt 7"};
        for (const auto& it : stuff)
            fprintf(gp, "%s\n", it.c_str());
        pclose(gp);
    }
}