//  2D Ising Model – Parallel Implementation (OpenMP)
//  Strategy 1 : Embarrassingly parallel over temperatures
//  Strategy 2 : Checkerboard (red-black) lattice decomposition
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <chrono>
#include <random>
#include <omp.h>

using namespace std;

static vector<mt19937> thread_rngs;

void init_thread_rngs(int n_threads)
{
    thread_rngs.resize(n_threads);
    mt19937 seeder(static_cast<unsigned>(time(nullptr)));
    for (int t = 0; t < n_threads; ++t)
        thread_rngs[t].seed(seeder());
}

inline double rand01(int tid)
{
    static uniform_real_distribution<double> d(0.0, 1.0);
    return d(thread_rngs[tid]);
}

inline int rand_L(int L, int tid)
{
    uniform_int_distribution<int> d(0, L - 1);
    return d(thread_rngs[tid]);
}

//  Lattice initialisation
void initialize_lattice(vector<vector<int>>& spin, int L, int tid)
{
    uniform_int_distribution<int> coin(0, 1);
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            spin[i][j] = (coin(thread_rngs[tid]) == 0) ? 1 : -1;
}

//  Energy change (periodic BC)
inline double compute_deltaE(const vector<vector<int>>& spin,
                             int i, int j, int L, double J)
{
    int up    = spin[(i - 1 + L) % L][j];
    int down  = spin[(i + 1)     % L][j];
    int left  = spin[i][(j - 1 + L) % L];
    int right = spin[i][(j + 1)     % L];
    return 2.0 * J * spin[i][j] * (up + down + left + right);
}

//  Strategy 1 – Random-site Metropolis sweep (serial within temp)
//  Used when parallelism is over temperature axis.
void metropolis_sweep_random(vector<vector<int>>& spin,
                             int L, double T, double J, int tid)
{
    const int N = L * L;
    for (int step = 0; step < N; ++step)
    {
        int i  = rand_L(L, tid);
        int j  = rand_L(L, tid);
        double dE = compute_deltaE(spin, i, j, L, J);
        if (dE < 0.0 || rand01(tid) < exp(-dE / T))
            spin[i][j] *= -1;
    }
}

//  Strategy 2 – Checkerboard (red-black) Metropolis sweep
//  All sites of the same colour are independent → parallelise rows.
//  colour = (i+j) % 2  (0 = red, 1 = black)
void metropolis_sweep_checkerboard(vector<vector<int>>& spin,
                                   int L, double T, double J)
{
    for (int colour = 0; colour < 2; ++colour)
    {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < L; ++i)
        {
            int tid = omp_get_thread_num();
            for (int j = (i + colour) % 2; j < L; j += 2)
            {
                double dE = compute_deltaE(spin, i, j, L, J);
                if (dE < 0.0 || rand01(tid) < exp(-dE / T))
                    spin[i][j] *= -1;
            }
        }
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
    return E / (L * L);
}
//  Run one temperature point
//  use_checkerboard == true  → Strategy 2 (intra-lattice OMP)
//  use_checkerboard == false → Strategy 1 (inter-temp OMP, random sweep)
void run_simulation(vector<vector<int>>& spin,
                    int L, double T, double J,
                    int equilibration_steps,
                    int measurement_steps,
                    bool use_checkerboard,
                    double& out_M, double& out_E,
                    double& out_Cv, double& out_Chi)
{
    int tid = omp_get_thread_num();

    auto sweep = [&]()
    {
        if (use_checkerboard)
            metropolis_sweep_checkerboard(spin, L, T, J);
        else
            metropolis_sweep_random(spin, L, T, J, tid);
    };

    for (int step = 0; step < equilibration_steps; ++step)
        sweep();

    double sumM = 0, sumM2 = 0, sumE = 0, sumE2 = 0;
    for (int step = 0; step < measurement_steps; ++step)
    {
        sweep();
        double m = fabs(calculate_magnetization(spin, L));
        double e = calculate_energy(spin, L, J);
        sumM += m;  sumM2 += m * m;
        sumE += e;  sumE2 += e * e;
    }

    out_M  = sumM  / measurement_steps;
    out_E  = sumE  / measurement_steps;
    double avgM2 = sumM2 / measurement_steps;
    double avgE2 = sumE2 / measurement_steps;
    out_Cv  = (avgE2 - out_E * out_E) / (T * T);
    out_Chi = (avgM2 - out_M * out_M) * (L * L) / T;
}

//  Scaling study: vary thread count for a fixed workload
void scaling_study(int L, double J,
                   int eq_steps, int meas_steps,
                   int max_threads)
{
    ofstream sf("speedup_data.txt");
    sf << "Threads Runtime_ms Speedup Efficiency\n";

    double T_ref   = 2.269; 
    double serial_time = 0.0;

    for (int nt = 1; nt <= max_threads; ++nt)
    {
        omp_set_num_threads(nt);
        init_thread_rngs(nt);

        const int n_temps = 41;  
        double T_min = 1.0;

        auto t0 = chrono::high_resolution_clock::now();

        #pragma omp parallel for schedule(dynamic)
        for (int ti = 0; ti < n_temps; ++ti)
        {
            double T = T_min + ti * 0.1;
            int tid  = omp_get_thread_num();
            vector<vector<int>> local_spin(L, vector<int>(L));
            initialize_lattice(local_spin, L, tid);
            double M, E, Cv, Chi;
            run_simulation(local_spin, L, T, J,
                           eq_steps, meas_steps,
                           false, M, E, Cv, Chi);
        }

        auto t1   = chrono::high_resolution_clock::now();
        double ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();

        if (nt == 1) serial_time = ms;
        double speedup    = serial_time / ms;
        double efficiency = speedup / nt * 100.0;

        sf << nt << " " << ms << " " << speedup << " " << efficiency << "\n";
        cout << "Threads=" << nt
             << "  Time=" << ms << " ms"
             << "  Speedup=" << speedup
             << "  Eff=" << efficiency << "%\n";
    }
    sf.close();
}

//  main
int main(int argc, char* argv[])
{
    const int    L                   = 50;
    const double J                   = 1.0;
    const int    equilibration_steps = 5000;
    const int    measurement_steps   = 10000;
    const double T_min               = 1.0;
    const double dT                  = 0.1;
    const int    n_temps             = 41;   // 1.0 to 5.0 in 0.1 steps

    // Detect available threads
    int max_threads = omp_get_max_threads();
    cout << "=== 2D Ising Model – OpenMP Parallel ===\n";
    cout << "Max threads available: " << max_threads << "\n";
    cout << "Lattice: " << L << "x" << L
         << "  Eq=" << equilibration_steps
         << "  Meas=" << measurement_steps << "\n\n";

    init_thread_rngs(max_threads);

    // ── Strategy 1: Parallelise over temperatures ─────────────
    {
        cout << "--- Strategy 1: Parallel over temperatures ---\n";
        omp_set_num_threads(max_threads);

        vector<double> res_M(n_temps), res_E(n_temps),
                       res_Cv(n_temps), res_Chi(n_temps);

        auto t0 = chrono::high_resolution_clock::now();

        #pragma omp parallel for schedule(dynamic)
        for (int ti = 0; ti < n_temps; ++ti)
        {
            double T  = T_min + ti * dT;
            int    tid = omp_get_thread_num();
            vector<vector<int>> local_spin(L, vector<int>(L));
            initialize_lattice(local_spin, L, tid);
            run_simulation(local_spin, L, T, J,
                           equilibration_steps, measurement_steps,
                           false,
                           res_M[ti], res_E[ti],
                           res_Cv[ti], res_Chi[ti]);
        }

        auto t1 = chrono::high_resolution_clock::now();
        long ms1 = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();

        ofstream of1("results_parallel_strategy1.txt");
        of1 << "Temperature Magnetization Energy HeatCapacity Susceptibility\n";
        for (int ti = 0; ti < n_temps; ++ti)
        {
            double T = T_min + ti * dT;
            of1 << T << " " << res_M[ti] << " " << res_E[ti]
                << " " << res_Cv[ti] << " " << res_Chi[ti] << "\n";
            cout << "T=" << T
                 << "  |M|="  << res_M[ti]
                 << "  E/N="  << res_E[ti]
                 << "  Cv="   << res_Cv[ti]
                 << "  Chi="  << res_Chi[ti] << "\n";
        }
        of1.close();
        cout << "\nStrategy 1 Runtime = " << ms1 << " ms\n\n";
        ofstream rt1("runtime_parallel_s1.txt"); rt1 << ms1; rt1.close();
    }

    // ── Strategy 2: Checkerboard within a single lattice ──────
    {
        cout << "--- Strategy 2: Checkerboard decomposition ---\n";
        omp_set_num_threads(max_threads);

        vector<double> res_M(n_temps), res_E(n_temps),
                       res_Cv(n_temps), res_Chi(n_temps);

        auto t0 = chrono::high_resolution_clock::now();

        for (int ti = 0; ti < n_temps; ++ti)
        {
            double T = T_min + ti * dT;
            vector<vector<int>> spin(L, vector<int>(L));
            initialize_lattice(spin, L, 0);   // main thread
            run_simulation(spin, L, T, J,
                           equilibration_steps, measurement_steps,
                           true,
                           res_M[ti], res_E[ti],
                           res_Cv[ti], res_Chi[ti]);
        }

        auto t1 = chrono::high_resolution_clock::now();
        long ms2 = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();

        ofstream of2("results_parallel_strategy2.txt");
        of2 << "Temperature Magnetization Energy HeatCapacity Susceptibility\n";
        for (int ti = 0; ti < n_temps; ++ti)
        {
            double T = T_min + ti * dT;
            of2 << T << " " << res_M[ti] << " " << res_E[ti]
                << " " << res_Cv[ti] << " " << res_Chi[ti] << "\n";
        }
        of2.close();
        cout << "Strategy 2 Runtime = " << ms2 << " ms\n\n";
        ofstream rt2("runtime_parallel_s2.txt"); rt2 << ms2; rt2.close();
    }

    cout << "--- Scaling study (Strategy 1 vs thread count) ---\n";
    scaling_study(L, J, equilibration_steps, measurement_steps, max_threads);

    return 0;
}
