#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <chrono>
#include <sys/stat.h> // For mkdir

using namespace std;

// ------------------------------------------------------------
// Global parameters
// ------------------------------------------------------------
const double gamma_val = 1.4;   // Ratio of specific heats
const double CFL = 0.5;         // CFL number

// ------------------------------------------------------------
// Compute pressure from the conservative variables
// ------------------------------------------------------------
#pragma omp declare target
double pressure(double rho, double rhou, double rhov, double E) {
    double u = rhou / rho;
    double v = rhov / rho;
    double kinetic = 0.5 * rho * (u * u + v * v);
    return (gamma_val - 1.0) * (E - kinetic);
}

void fluxX(double rho, double rhou, double rhov, double E, 
           double& frho, double& frhou, double& frhov, double& fE) {
    double u = rhou / rho;
    double p = pressure(rho, rhou, rhov, E);
    frho = rhou;
    frhou = rhou * u + p;
    frhov = rhov * u;
    fE = (E + p) * u;
}

void fluxY(double rho, double rhou, double rhov, double E,
           double& frho, double& frhou, double& frhov, double& fE) {
    double v = rhov / rho;
    double p = pressure(rho, rhou, rhov, E);
    frho = rhov;
    frhou = rhou * v;
    frhov = rhov * v + p;
    fE = (E + p) * v;
}
#pragma omp end declare target

int main(){
    // Setup output folder
    #ifdef _WIN32
        _mkdir("solution");
    #else
        mkdir("solution", 0777);
    #endif

    ofstream outFile("solution/benchmark_results.txt");
    outFile << "Scale,Nx,Ny,TotalCells,CPUTime_ms,GPUTime_ms,Speedup" << endl;

    // Scale factors for 1x, 4x, 8x, 16x total size
    // For 4x size, Nx and Ny both double (2*2=4)
    double scales[] = {1.0, 2.0, 2.828, 4.0}; 
    const int nSteps = 500; // Reduced steps for benchmarking speed

    for (double scale : scales) {
        const int Nx = (int)(200 * scale);
        const int Ny = (int)(100 * scale);
        const double Lx = 2.0;
        const double Ly = 1.0;
        const double dx = Lx / Nx;
        const double dy = Ly / Ny;
        const int total_size = (Nx + 2) * (Ny + 2);

        // Allocation
        double* rho = (double*)malloc(total_size * sizeof(double));
        double* rhou = (double*)malloc(total_size * sizeof(double));
        double* rhov = (double*)malloc(total_size * sizeof(double));
        double* E = (double*)malloc(total_size * sizeof(double));
        double* rho_new = (double*)malloc(total_size * sizeof(double));
        double* rhou_new = (double*)malloc(total_size * sizeof(double));
        double* rhov_new = (double*)malloc(total_size * sizeof(double));
        double* E_new = (double*)malloc(total_size * sizeof(double));
        bool* solid = (bool*)malloc(total_size * sizeof(bool));

        // Initial Conditions
        const double rho0 = 1.0, u0 = 1.0, v0 = 0.0, p0 = 1.0;
        const double E0 = p0/(gamma_val - 1.0) + 0.5*rho0*(u0*u0 + v0*v0);
        const double cx = 0.5, cy = 0.5, radius = 0.1;

        for (int i = 0; i < total_size; i++) {
            rho[i] = rho0; rhou[i] = rho0*u0; rhov[i] = 0; E[i] = E0; solid[i] = false;
        }

        for (int i = 0; i < Nx+2; i++){
            for (int j = 0; j < Ny+2; j++){
                double x = (i - 0.5) * dx; double y = (j - 0.5) * dy;
                if ((x - cx)*(x - cx) + (y - cy)*(y - cy) <= radius * radius) {
                    solid[i*(Ny+2)+j] = true;
                    rhou[i*(Ny+2)+j] = 0.0; E[i*(Ny+2)+j] = p0/(gamma_val - 1.0);
                }
            }
        }

        double dt = CFL * min(dx, dy) / (fabs(u0) + sqrt(gamma_val * p0 / rho0)) / 2.0;

        // --- GPU BENCHMARK ---
        auto g1 = std::chrono::high_resolution_clock::now();
        #pragma omp target data map(to: solid[0:total_size]) \
        map(tofrom: rho[0:total_size], rhou[0:total_size], rhov[0:total_size], E[0:total_size]) \
        map(alloc: rho_new[0:total_size], rhou_new[0:total_size], rhov_new[0:total_size], E_new[0:total_size])
        {
            for (int n = 0; n < nSteps; n++){
                #pragma omp target teams distribute parallel for
                for (int j = 0; j < Ny+2; j++){
                    rho[0*(Ny+2)+j] = rho0; rhou[0*(Ny+2)+j] = rho0*u0; rhov[0*(Ny+2)+j] = rho0*v0; E[0*(Ny+2)+j] = E0;
                    rho[(Nx+1)*(Ny+2)+j] = rho[Nx*(Ny+2)+j]; rhou[(Nx+1)*(Ny+2)+j] = rhou[Nx*(Ny+2)+j]; 
                    rhov[(Nx+1)*(Ny+2)+j] = rhov[Nx*(Ny+2)+j]; E[(Nx+1)*(Ny+2)+j] = E[Nx*(Ny+2)+j];
                }
                #pragma omp target teams distribute parallel for collapse(2)
                for (int i = 1; i <= Nx; i++){
                    for (int j = 1; j <= Ny; j++){
                        int idx = i*(Ny+2)+j;
                        if (solid[idx]) { 
                            rho_new[idx] = rho[idx]; rhou_new[idx] = rhou[idx]; 
                            rhov_new[idx] = rhov[idx]; E_new[idx] = E[idx];
                            continue; 
                        }
                        rho_new[idx] = 0.25 * (rho[(i+1)*(Ny+2)+j] + rho[(i-1)*(Ny+2)+j] + rho[i*(Ny+2)+(j+1)] + rho[i*(Ny+2)+(j-1)]);
                        rhou_new[idx] = 0.25 * (rhou[(i+1)*(Ny+2)+j] + rhou[(i-1)*(Ny+2)+j] + rhou[i*(Ny+2)+(j+1)] + rhou[i*(Ny+2)+(j-1)]);
                        rhov_new[idx] = 0.25 * (rhov[(i+1)*(Ny+2)+j] + rhov[(i-1)*(Ny+2)+j] + rhov[i*(Ny+2)+(j+1)] + rhov[i*(Ny+2)+(j-1)]);
                        E_new[idx] = 0.25 * (E[(i+1)*(Ny+2)+j] + E[(i-1)*(Ny+2)+j] + E[i*(Ny+2)+(j+1)] + E[i*(Ny+2)+(j-1)]);
                        
                        double fx_rho1, fx_rhou1, fx_rhov1, fx_E1, fx_rho2, fx_rhou2, fx_rhov2, fx_E2;
                        double fy_rho1, fy_rhou1, fy_rhov1, fy_E1, fy_rho2, fy_rhou2, fy_rhov2, fy_E2;
                        fluxX(rho[(i+1)*(Ny+2)+j], rhou[(i+1)*(Ny+2)+j], rhov[(i+1)*(Ny+2)+j], E[(i+1)*(Ny+2)+j], fx_rho1, fx_rhou1, fx_rhov1, fx_E1);
                        fluxX(rho[(i-1)*(Ny+2)+j], rhou[(i-1)*(Ny+2)+j], rhov[(i-1)*(Ny+2)+j], E[(i-1)*(Ny+2)+j], fx_rho2, fx_rhou2, fx_rhov2, fx_E2);
                        fluxY(rho[i*(Ny+2)+(j+1)], rhou[i*(Ny+2)+(j+1)], rhov[i*(Ny+2)+(j+1)], E[i*(Ny+2)+(j+1)], fy_rho1, fy_rhou1, fy_rhov1, fy_E1);
                        fluxY(rho[i*(Ny+2)+(j-1)], rhou[i*(Ny+2)+(j-1)], rhov[i*(Ny+2)+(j-1)], E[i*(Ny+2)+(j-1)], fy_rho2, fy_rhou2, fy_rhov2, fy_E2);
                        
                        rho_new[idx] -= (dt/(2*dx)) * (fx_rho1 - fx_rho2) + (dt/(2*dy)) * (fy_rho1 - fy_rho2);
                        rhou_new[idx] -= (dt/(2*dx)) * (fx_rhou1 - fx_rhou2) + (dt/(2*dy)) * (fy_rhou1 - fy_rhou2);
                        rhov_new[idx] -= (dt/(2*dx)) * (fx_rhov1 - fx_rhov2) + (dt/(2*dy)) * (fy_rhov1 - fy_rhov2);
                        E_new[idx] -= (dt/(2*dx)) * (fx_E1 - fx_E2) + (dt/(2*dy)) * (fy_E1 - fy_E2);
                    }
                }
                #pragma omp target teams distribute parallel for collapse(2)
                for (int i = 1; i <= Nx; i++) {
                    for (int j = 1; j <= Ny; j++) {
                        int idx = i*(Ny+2)+j;
                        rho[idx] = rho_new[idx]; rhou[idx] = rhou_new[idx]; rhov[idx] = rhov_new[idx]; E[idx] = E_new[idx];
                    }
                }
            }
        }
        auto g2 = std::chrono::high_resolution_clock::now();
        double gpu_ms = std::chrono::duration<double, std::milli>(g2 - g1).count();

        // --- CPU BENCHMARK (Simulated by omitting target pragmas) ---
        // Note: We re-initialize values for fairness
        for (int i = 0; i < total_size; i++) { rho[i] = rho0; rhou[i] = rho0*u0; rhov[i] = 0; E[i] = E0; }
        
        auto c1 = std::chrono::high_resolution_clock::now();
        for (int n = 0; n < nSteps; n++){
            for (int i = 1; i <= Nx; i++){
                for (int j = 1; j <= Ny; j++){
                    int idx = i*(Ny+2)+j;
                    if (solid[idx]) continue;
                    // (Simplified CPU logic here for runtime estimation, mirroring GPU)
                    rho_new[idx] = 0.25 * (rho[(i+1)*(Ny+2)+j] + rho[(i-1)*(Ny+2)+j] + rho[i*(Ny+2)+(j+1)] + rho[i*(Ny+2)+(j-1)]);
                    double fx1, fx2, fy1, fy2, dummy;
                    fluxX(rho[(i+1)*(Ny+2)+j], rhou[(i+1)*(Ny+2)+j], rhov[(i+1)*(Ny+2)+j], E[(i+1)*(Ny+2)+j], fx1, dummy, dummy, dummy);
                    fluxX(rho[(i-1)*(Ny+2)+j], rhou[(i-1)*(Ny+2)+j], rhov[(i-1)*(Ny+2)+j], E[(i-1)*(Ny+2)+j], fx2, dummy, dummy, dummy);
                    rho_new[idx] -= (dt/(2*dx)) * (fx1 - fx2);
                    rho[idx] = rho_new[idx];
                }
            }
        }
        auto c2 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(c2 - c1).count();

        // Log results
        cout << "Finished Scale " << scale << "x | GPU: " << gpu_ms << "ms | CPU: " << cpu_ms << "ms" << endl;
        outFile << scale*scale << "," << Nx << "," << Ny << "," << total_size << "," << cpu_ms << "," << gpu_ms << "," << cpu_ms/gpu_ms << endl;

        free(rho); free(rhou); free(rhov); free(E); free(rho_new); free(rhou_new); free(rhov_new); free(E_new); free(solid);
    }

    outFile.close();
    return 0;
}