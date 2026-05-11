//  2D Ising Model – Serial Implementation (Metropolis MC)
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <chrono>
#include <numeric>
#include <random>

using namespace std;
static mt19937 rng(static_cast<unsigned>(time(nullptr)));
static uniform_real_distribution<double> dist01(0.0, 1.0);
static uniform_int_distribution<int>     dist_spin(0, 1);

//  Lattice initialisation
void initialize_lattice(vector<vector<int>>& spin, int L)
{
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            spin[i][j] = (dist_spin(rng) == 0) ? 1 : -1;
}

//  Energy change for a single spin flip (periodic BC)
inline double compute_deltaE(const vector<vector<int>>& spin,
                             int i, int j, int L, double J)
{
    int up    = spin[(i - 1 + L) % L][j];
    int down  = spin[(i + 1)     % L][j];
    int left  = spin[i][(j - 1 + L) % L];
    int right = spin[i][(j + 1)     % L];
    return 2.0 * J * spin[i][j] * (up + down + left + right);
}

//  One Metropolis sweep  (L*L random-site updates)
void metropolis_update(vector<vector<int>>& spin,
                       int L, double T, double J)
{
    uniform_int_distribution<int> rand_L(0, L - 1);
    const int N = L * L;

    for (int step = 0; step < N; ++step)
    {
        int i = rand_L(rng);
        int j = rand_L(rng);
        double dE = compute_deltaE(spin, i, j, L, J);

        if (dE < 0.0 || dist01(rng) < exp(-dE / T))
            spin[i][j] *= -1;
    }
}

//  Observables
double calculate_magnetization(const vector<vector<int>>& spin, int L)
{
    long M = 0;
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            M += spin[i][j];
    return static_cast<double>(M) / (L * L);
}

double calculate_energy(const vector<vector<int>>& spin,
                        int L, double J)
{
    double E = 0.0;
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
        {
            int right = spin[i][(j + 1) % L];
            int down  = spin[(i + 1) % L][j];
            E += -J * spin[i][j] * (right + down);
        }
    return E / (L * L);          // energy per spin
}

//  Write a spin snapshot for visualisation
void write_snapshot(const vector<vector<int>>& spin, int L,
                    double T, const string& filename)
{
    ofstream f(filename);
    f << "# Spin snapshot at T=" << T << "\n";
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
            f << spin[i][j] << " ";
        f << "\n";
    }
}

//  Single-temperature simulation
void run_simulation(vector<vector<int>>& spin,
                    int L, double T, double J,
                    int equilibration_steps,
                    int measurement_steps,
                    ofstream& outfile,
                    bool save_snapshot = false)
{
    // Equilibration
    for (int step = 0; step < equilibration_steps; ++step)
        metropolis_update(spin, L, T, J);

    // Measurement
    double sumM = 0.0, sumM2 = 0.0;
    double sumE = 0.0, sumE2 = 0.0;

    for (int step = 0; step < measurement_steps; ++step)
    {
        metropolis_update(spin, L, T, J);
        double m = fabs(calculate_magnetization(spin, L));
        double e = calculate_energy(spin, L, J);
        sumM  += m;   sumM2 += m * m;
        sumE  += e;   sumE2 += e * e;
    }

    double avgM  = sumM  / measurement_steps;
    double avgE  = sumE  / measurement_steps;
    double avgM2 = sumM2 / measurement_steps;
    double avgE2 = sumE2 / measurement_steps;

    // Heat capacity and magnetic susceptibility per spin
    double Cv  = (avgE2 - avgE * avgE) / (T * T);
    double Chi = (avgM2 - avgM * avgM) * (L * L) / T;

    outfile << T << " " << avgM << " " << avgE
            << " " << Cv << " " << Chi << "\n";

    cout << "T=" << T
         << "  |M|="   << avgM
         << "  E/N="   << avgE
         << "  Cv="    << Cv
         << "  Chi="   << Chi << "\n";

    // Save snapshot near critical temperature and at extremes
    if (save_snapshot)
    {
        string fname = "snapshot_T" + to_string(T).substr(0,4) + ".txt";
        write_snapshot(spin, L, T, fname);
    }
}

//  main
int main()
{
    const int    L                   = 50;
    const double J                   = 1.0;
    const int    equilibration_steps = 5000;
    const int    measurement_steps   = 10000;

    const double T_min  = 1.0;
    const double T_max  = 5.0;
    const double dT     = 0.1;
    const double T_crit = 2.269;

    vector<vector<int>> spin(L, vector<int>(L));

    // Prepare output file
    ofstream outfile("results_serial.txt");
    outfile << "Temperature Magnetization Energy HeatCapacity Susceptibility\n";

    cout << "=== 2D Ising Model – Serial ===\n";
    cout << "Lattice: " << L << "x" << L
         << "  Eq=" << equilibration_steps
         << "  Meas=" << measurement_steps << "\n\n";

    auto global_start = chrono::high_resolution_clock::now();

    // Temperatures to snapshot
    // Low, critical, high
    vector<double> snap_temps = {1.0, T_crit, 4.0};

    for (double T = T_min; T <= T_max + 1e-9; T += dT)
    {
        // Fresh random lattice for each temperature
        initialize_lattice(spin, L);

        // Save snapshot near specific temps?
        bool snap = false;
        for (double ts : snap_temps)
            if (fabs(T - ts) < dT / 2.0) snap = true;

        run_simulation(spin, L, T, J,
                       equilibration_steps, measurement_steps,
                       outfile, snap);
    }

    auto global_stop = chrono::high_resolution_clock::now();
    auto total_ms = chrono::duration_cast<chrono::milliseconds>
                    (global_stop - global_start).count();

    cout << "\nTotal Runtime = " << total_ms << " ms\n";

    // Write runtime to a file for comparison with parallel
    ofstream rt("runtime_serial.txt");
    rt << total_ms << "\n";
    rt.close();

    outfile.close();
    return 0;
}
