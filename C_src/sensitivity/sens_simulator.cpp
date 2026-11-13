// Forward-sensitivity FTACV simulator (single electron transfer, Butler-Volmer, no dispersion).
//
// Standalone module `SurfaceODESensitivity`. It does NOT replace or modify SurfaceODESolver: it
// re-states the right hand side of `single_e` (headers/functions.cpp) as a pure function of an
// explicit parameter array p, so that CVODES can differentiate it, and returns the total current
// together with dI/dp at every requested time.
//
// Differences from ODEsimulate (deliberate):
//   * f(t,y,p) is pure (single_e writes kox/kred/Cdlp back into the parameter map)
//   * the Jacobian is exact. The one in functions.cpp omits the alpha factors in dtheta/dI and the
//     dCdlp/dI term, and reads stale kox/kred; harmless for the state solve, not for sensitivities
//   * index 0 of the output is y(0) rather than 0
//   * only Cdl > 0 (the capacitive-ODE branch) and Butler-Volmer kinetics are supported
//
// State y=[I, theta]. Nondimensional parameters, exactly as handed to ODEsimulate.
//
// Accuracy note: the default BDF order (max_order=0 -> 5) is what the sensitivity results were verified with. With
// sensitivities included in the error test the current is accurate (1.5e-8 vs a converged reference at rtol=atol=1e-8).
// A forward-only call (empty sens_names) with order 5 is NOT: the error test does not see theta errors amplified into
// the current by gamma*(kox+kred); pass max_order=2 for forward-only use (see the same fix in ODE_simulator_sundials.cpp).
#include <algorithm>
#include <atomic>
#include <cmath>
#include <thread>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <cvodes/cvodes.h>
#include <nvector/nvector_serial.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sundials/sundials_types.h>
#include <sundials/sundials_config.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;

// SUNDIALS 6.x (e.g. ARC's SUNDIALS/6.3.0) and 7.x differ in the three places below.
#if defined(SUNDIALS_VERSION_MAJOR) && SUNDIALS_VERSION_MAJOR >= 7
  #define SENS_COMM_NULL SUN_COMM_NULL
  #define SENS_SILENCE_ERRORS(ctx, mem) SUNContext_ClearErrHandlers(ctx)
#else
  #define SENS_COMM_NULL NULL
  static void quiet_err_handler(int, const char*, const char*, char*, void*) {}
  #define SENS_SILENCE_ERRORS(ctx, mem) CVodeSetErrHandlerFn(mem, quiet_err_handler, nullptr)
#endif

// differentiable parameters, in the order they live in p[]
static const char* PNAMES[] = {"E0", "k0", "gamma", "Cdl", "alpha", "Ru", "CdlE1", "CdlE2", "CdlE3"};
enum { iE0, ik0, igamma, iCdl, ialpha, iRu, iC1, iC2, iC3, NP };

struct Data {
    double p[NP];                       // CVODES perturbs this array in place for difference quotients
    double E_start, E_reverse, v, omega, delta_E, phase, cap_phase, tr;
    double rtol_w = 0, atol_w = 0, atol_th_w = 0;   // tolerances for the error-weight function
    mutable double last_kk = 0;         // kox+kred at the most recent rhs call (used by the error-weight function)
    int plist[NP];                      // indices into p[] of the parameters we differentiate with respect to
};

static inline double E_of_t(const Data& d, double t, double phase) {
    double Edc = (t < d.tr) ? d.E_start + d.v * t : d.E_reverse - d.v * (t - d.tr);
    return Edc + d.delta_E * std::sin(d.omega * t + phase);
}
static inline double dE_of_t(const Data& d, double t, double phase) {
    double Edc = (t < d.tr) ? d.v : -d.v;
    return Edc + d.delta_E * d.omega * std::cos(d.omega * t + phase);
}

// everything the RHS and Jacobian share
struct Terms { double kox, kred, f2, Cdlp, dCdlp_dI, cap_dE; };
static Terms terms(const Data& d, double t, double I, double theta) {
    const double* p = d.p;
    Terms T;
    double Er = E_of_t(d, t, d.phase) - I * p[iRu];
    double cap_Er = E_of_t(d, t, d.cap_phase) - I * p[iRu];
    T.cap_dE = dE_of_t(d, t, d.cap_phase);
    T.kred = p[ik0] * std::exp(-p[ialpha] * (Er - p[iE0]));
    T.kox = p[ik0] * std::exp((1 - p[ialpha]) * (Er - p[iE0]));
    T.f2 = T.kox * (1 - theta) - T.kred * theta;
    double c2 = cap_Er * cap_Er;
    T.Cdlp = p[iCdl] * (1 + p[iC1] * cap_Er + p[iC2] * c2 + p[iC3] * c2 * cap_Er);
    double dpoly = p[iC1] + 2 * p[iC2] * cap_Er + 3 * p[iC3] * c2;   // d(poly)/d(cap_Er)
    T.dCdlp_dI = p[iCdl] * dpoly * (-p[iRu]);
    return T;
}

static int rhs(sunrealtype t, N_Vector y, N_Vector ydot, void* ud) {
    const Data& d = *static_cast<Data*>(ud);
    double I = NV_Ith_S(y, 0), th = NV_Ith_S(y, 1);
    Terms T = terms(d, t, I, th);
    // algebraically identical to dIdt in single_e: -(1/(Ru Cdlp)) (I - gamma f2 - Cdlp cap_dE)
    NV_Ith_S(ydot, 0) = T.cap_dE / d.p[iRu] - (I - d.p[igamma] * T.f2) / (d.p[iRu] * T.Cdlp);
    NV_Ith_S(ydot, 1) = T.f2;
    d.last_kk = T.kox + T.kred;
    return 0;
}

static int jac(sunrealtype t, N_Vector y, N_Vector, SUNMatrix J, void* ud, N_Vector, N_Vector, N_Vector) {
    const Data& d = *static_cast<Data*>(ud);
    const double* p = d.p;
    double I = NV_Ith_S(y, 0), th = NV_Ith_S(y, 1);
    Terms T = terms(d, t, I, th);
    double df2_dI = -(1 - p[ialpha]) * p[iRu] * T.kox * (1 - th) - p[ialpha] * p[iRu] * T.kred * th;
    double df2_dth = -T.kox - T.kred;
    double num = I - p[igamma] * T.f2, den = p[iRu] * T.Cdlp;
    SM_ELEMENT_D(J, 0, 0) = -(1 - p[igamma] * df2_dI) / den + num / (den * T.Cdlp) * T.dCdlp_dI;
    SM_ELEMENT_D(J, 0, 1) = p[igamma] * df2_dth / den;
    SM_ELEMENT_D(J, 1, 0) = df2_dI;
    SM_ELEMENT_D(J, 1, 1) = df2_dth;
    return 0;
}

// Error weights that account for how theta errors enter the measured current. I relaxes (rate 1/(Ru Cdlp)) to
// Cdlp*dE/dt + gamma*theta', and d(theta')/d(theta) = -(kox+kred), so an error dtheta becomes a current error of
// ~gamma*(kox+kred)*dtheta. The standard weights only compare dtheta with rtol*|theta|+atol, which can be 1e4x too lax.
static int efun_current(N_Vector y, N_Vector ewt, void* ud) {
    const Data& d = *static_cast<Data*>(ud);
    double wI = 1.0 / (d.rtol_w * std::fabs(NV_Ith_S(y, 0)) + d.atol_w);
    double wth = 1.0 / (d.rtol_w * std::fabs(NV_Ith_S(y, 1)) + d.atol_th_w);
    double wth_cur = d.p[igamma] * d.last_kk * wI;           // error in theta that moves the current by 1 tolerance unit
    NV_Ith_S(ewt, 0) = wI;
    NV_Ith_S(ewt, 1) = std::fmax(wth, wth_cur);
    return 0;
}

// Sensitivity right hand side: S_i' = (df/dy) S_i + df/dp_{plist[i]}   (all partials analytic)
//   f1 = cap_dE/Ru - (I - gamma f2)/(Ru Cdlp),   f2 = kox (1-theta) - kred theta
static int rhsS(int Ns, sunrealtype t, N_Vector y, N_Vector, N_Vector* yS, N_Vector* ySdot, void* ud, N_Vector, N_Vector) {
    const Data& d = *static_cast<Data*>(ud);
    const double* p = d.p;
    const double I = NV_Ith_S(y, 0), th = NV_Ith_S(y, 1);
    const Terms T = terms(d, t, I, th);
    const double a = p[ialpha], Ru = p[iRu], g = p[igamma];
    const double cap_Er = E_of_t(d, t, d.cap_phase) - I * Ru;
    const double Er_E0 = E_of_t(d, t, d.phase) - I * Ru - p[iE0];
    const double dcap = p[iCdl] * (p[iC1] + 2 * p[iC2] * cap_Er + 3 * p[iC3] * cap_Er * cap_Er);   // dCdlp/dcap_Er
    const double dfdEr = (1 - a) * T.kox * (1 - th) + a * T.kred * th;                         // df2/dEr
    const double df2_dI = -Ru * dfdEr, df2_dth = -T.kox - T.kred;
    const double num = I - g * T.f2, den = Ru * T.Cdlp;
    const double J00 = -(1 - g * df2_dI) / den + num / (den * T.Cdlp) * T.dCdlp_dI;
    const double J01 = g * df2_dth / den;
    for (int i = 0; i < Ns; ++i) {
        double df1 = 0, df2 = 0;
        switch (d.plist[i]) {
            case iE0:    df2 = -(1 - a) * T.kox * (1 - th) - a * T.kred * th;  df1 = g * df2 / den; break;
            case ik0:    df2 = T.f2 / p[ik0];                                    df1 = g * df2 / den; break;
            case igamma: df1 = T.f2 / den; break;
            case ialpha: df2 = -Er_E0 * T.f2;                                    df1 = g * df2 / den; break;
            case iRu:    df2 = -I * dfdEr;
                         df1 = -T.cap_dE / (Ru * Ru) + num / (Ru * den) + g * df2 / den + num / (den * T.Cdlp) * (-I * dcap); break;
            case iCdl:   df1 = num / (den * T.Cdlp) * (T.Cdlp / p[iCdl]); break;
            case iC1:    df1 = num / (den * T.Cdlp) * p[iCdl] * cap_Er; break;
            case iC2:    df1 = num / (den * T.Cdlp) * p[iCdl] * cap_Er * cap_Er; break;
            case iC3:    df1 = num / (den * T.Cdlp) * p[iCdl] * cap_Er * cap_Er * cap_Er; break;
        }
        const double S0 = NV_Ith_S(yS[i], 0), S1 = NV_Ith_S(yS[i], 1);
        NV_Ith_S(ySdot[i], 0) = J00 * S0 + J01 * S1 + df1;
        NV_Ith_S(ySdot[i], 1) = df2_dI * S0 + df2_dth * S1 + df2;
    }
    return 0;
}

static double need(const std::unordered_map<std::string, double>& m, const char* k) {
    auto it = m.find(k);
    if (it == m.end()) throw std::invalid_argument(std::string("missing parameter: ") + k);
    return it->second;
}
static double opt(const std::unordered_map<std::string, double>& m, const char* k, double dflt) {
    auto it = m.find(k);
    return it == m.end() ? dflt : it->second;
}
#define CHK(call, name) do { int r_ = (call); if (r_ < 0) throw std::runtime_error(std::string(name) + " failed, retval=" + std::to_string(r_)); } while (0)

struct Opts {
    double rtol = 1e-8, atol = 1e-8; long max_steps = 1000000; bool analytic_sens = true, analytic_jac = true;
    double atol_theta = -1.0; int max_err_test_fails = 50; double max_step = 0.0; int max_order = 0; bool current_weighted = false; bool sens_errcon = true;
};
struct Out { int nt = 0, Ns = 0; std::vector<double> I, th, SI, Sth; };   // SI, Sth are [nt, Ns] row-major

// Pure C++ (no Python objects), so it can run on several threads: every call owns its SUNContext, solver memory and Data.
static Out solve_core(const std::vector<double>& times, const std::unordered_map<std::string, double>& params,
                      const std::vector<std::string>& sens_names, const Opts& o) {
    const double rtol = o.rtol, atol = o.atol, atol_theta = o.atol_theta; const long max_steps = o.max_steps;
    const bool analytic_sens = o.analytic_sens, analytic_jac = o.analytic_jac, current_weighted = o.current_weighted;
    const int max_err_test_fails = o.max_err_test_fails, max_order = o.max_order; const double max_step = o.max_step;
    Data d;
    for (int i = 0; i < NP; ++i) d.p[i] = (i >= iC1) ? opt(params, PNAMES[i], 0.0) : need(params, PNAMES[i]);
    if (d.p[iCdl] <= 0) throw std::invalid_argument("Cdl must be > 0 (capacitive ODE branch only)");
    d.E_start = need(params, "E_start"); d.E_reverse = need(params, "E_reverse");
    d.v = need(params, "v"); d.omega = need(params, "omega"); d.delta_E = need(params, "delta_E");
    d.phase = need(params, "phase"); d.cap_phase = opt(params, "cap_phase", d.phase); d.tr = need(params, "tr");
    double theta0 = opt(params, "theta", 0.0);

    std::vector<int> plist;
    for (auto& n : sens_names) {
        int idx = -1;
        for (int i = 0; i < NP; ++i) if (n == PNAMES[i]) idx = i;
        if (idx < 0) throw std::invalid_argument("unknown sensitivity parameter: " + n);
        plist.push_back(idx);
    }
    for (size_t i = 0; i < plist.size() && i < NP; ++i) d.plist[i] = plist[i];
    const int Ns = (int)plist.size(), nt = (int)times.size(), NEQ = 2;
    if (nt < 2) throw std::invalid_argument("need at least two times");

    SUNContext ctx; CHK(SUNContext_Create(SENS_COMM_NULL, &ctx), "SUNContext_Create");
    N_Vector y = N_VNew_Serial(NEQ, ctx), abstol = N_VNew_Serial(NEQ, ctx);
    N_Vector* yS = Ns ? N_VCloneVectorArray(Ns, y) : nullptr;
    NV_Ith_S(y, 0) = d.p[iCdl] * dE_of_t(d, 0.0, d.phase);   // as ODEsimulate: uses phase, not cap_phase
    NV_Ith_S(y, 1) = theta0;
    NV_Ith_S(abstol, 0) = atol; NV_Ith_S(abstol, 1) = (atol_theta > 0) ? atol_theta : atol;

    void* mem = CVodeCreate(CV_BDF, ctx);
    SENS_SILENCE_ERRORS(ctx, mem);   // failures are reported through the exception (with step/state details), not printed by SUNDIALS
    SUNMatrix A = SUNDenseMatrix(NEQ, NEQ, ctx);
    SUNLinearSolver LS = SUNLinSol_Dense(y, A, ctx);
    std::vector<double> pbar(Ns), atolS(Ns);
    Out out; out.nt = nt; out.Ns = Ns; out.I.assign(nt, 0.0); out.th.assign(nt, 0.0); out.SI.assign((size_t)nt * Ns, 0.0); out.Sth.assign((size_t)nt * Ns, 0.0);
    double *pI = out.I.data(), *pth = out.th.data(), *pSI = out.SI.data(), *pSth = out.Sth.data();
    auto record = [&](int k) {
        pI[k] = NV_Ith_S(y, 0); pth[k] = NV_Ith_S(y, 1);
        for (int s = 0; s < Ns; ++s) { pSI[k * Ns + s] = NV_Ith_S(yS[s], 0); pSth[k * Ns + s] = NV_Ith_S(yS[s], 1); }
    };
    std::string err;
    try {
        CHK(CVodeInit(mem, rhs, 0.0, y), "CVodeInit");
        d.rtol_w = rtol; d.atol_w = atol; d.atol_th_w = (atol_theta > 0) ? atol_theta : atol;
        if (current_weighted) CHK(CVodeWFtolerances(mem, efun_current), "CVodeWFtolerances");   // ewt fn gets the CVODE user_data
        else CHK(CVodeSVtolerances(mem, rtol, abstol), "CVodeSVtolerances");
        CHK(CVodeSetUserData(mem, &d), "CVodeSetUserData");
        CHK(CVodeSetMaxNumSteps(mem, max_steps), "CVodeSetMaxNumSteps");
        CHK(CVodeSetLinearSolver(mem, LS, A), "CVodeSetLinearSolver");
        // Bursts of consecutive step rejections happen as the fast modes re-equilibrate (the Jacobian eigenvalues
        // are all real and negative, so this is not an instability); CVODE's default limit of 7 aborts on them.
        CHK(CVodeSetMaxErrTestFails(mem, max_err_test_fails), "CVodeSetMaxErrTestFails");
        if (max_step > 0) CHK(CVodeSetMaxStep(mem, max_step), "CVodeSetMaxStep");
        if (max_order > 0) CHK(CVodeSetMaxOrd(mem, max_order), "CVodeSetMaxOrd");
        if (analytic_jac) CHK(CVodeSetJacFn(mem, jac), "CVodeSetJacFn");
        if (Ns) {
            // initial sensitivities: y0 depends on Cdl only (I0 = Cdl * dE(0)); theta0 is fixed
            for (int s = 0; s < Ns; ++s) {
                N_VConst(0.0, yS[s]);
                if (plist[s] == iCdl) NV_Ith_S(yS[s], 0) = dE_of_t(d, 0.0, d.phase);
                pbar[s] = std::fabs(d.p[plist[s]]) > 0 ? std::fabs(d.p[plist[s]]) : 1.0;
                atolS[s] = atol / pbar[s];   // sensitivities are scaled by pbar, as in the CVODES guide
            }
            CHK(CVodeSensInit(mem, Ns, CV_SIMULTANEOUS, analytic_sens ? rhsS : nullptr, yS), "CVodeSensInit");  // nullptr => CVODES difference quotients of rhs
            CHK(CVodeSetSensParams(mem, d.p, pbar.data(), plist.data()), "CVodeSetSensParams");
            CHK(CVodeSensEEtolerances(mem), "CVodeSensEEtolerances");
            CHK(CVodeSetSensErrCon(mem, o.sens_errcon ? SUNTRUE : SUNFALSE), "CVodeSetSensErrCon");
        }
        record(0);
        sunrealtype tret;
        // dE/dt jumps at the scan reversal t=tr, so f(t,y) is discontinuous there. Stop exactly at tr and
        // restart (state and sensitivities) instead of letting the BDF history straddle the jump.
        bool crossed = !(d.tr > 0 && d.tr < times.back());
        const double eps_t = 1e-12 * std::fabs(d.tr);
        auto restart = [&]() {
            CHK(CVodeReInit(mem, d.tr, y), "CVodeReInit");
            if (Ns) CHK(CVodeSensReInit(mem, CV_SIMULTANEOUS, yS), "CVodeSensReInit");
            // SUNDIALS 6.x keeps the (now reached) stop time armed after ReInit and then rejects it as lying behind t (7.x clears it).
            // CVodeClearStopTime is not in every 6.x release, so move the stop time beyond the end of the integration instead.
            CHK(CVodeSetStopTime(mem, 2.0 * times.back() + 1.0), "CVodeSetStopTime");
            crossed = true;
        };
        if (!crossed) CHK(CVodeSetStopTime(mem, d.tr), "CVodeSetStopTime");
        for (int k = 1; k < nt; ++k) {
            if (!crossed && times[k] > d.tr + eps_t) {
                CHK(CVode(mem, d.tr, y, &tret, CV_NORMAL), "CVode (to tr)");
                if (Ns) CHK(CVodeGetSens(mem, &tret, yS), "CVodeGetSens");
                restart();
            }
            int rv = CVode(mem, times[k], y, &tret, CV_NORMAL);
            if (rv < 0) {
                sunrealtype tc, hl; long nst, netf, ncf; CVodeGetCurrentTime(mem, &tc); CVodeGetLastStep(mem, &hl);
                CVodeGetNumSteps(mem, &nst); CVodeGetNumErrTestFails(mem, &netf); CVodeGetNumNonlinSolvConvFails(mem, &ncf);
                N_Vector ew = N_VClone(y); CVodeGetErrWeights(mem, ew);
                struct EwFree { N_Vector v; ~EwFree() { N_VDestroy(v); } } ewfree{ew};
                throw std::runtime_error("CVode failed rv=" + std::to_string(rv) + " at t=" + std::to_string(tc) + " h=" + std::to_string(hl) +
                    " steps=" + std::to_string(nst) + " errtestfails=" + std::to_string(netf) + " nlconvfails=" + std::to_string(ncf) +
                    " I=" + std::to_string(NV_Ith_S(y,0)) + " theta=" + std::to_string(NV_Ith_S(y,1)) +
                    " ewt=[" + std::to_string(NV_Ith_S(ew,0)) + "," + std::to_string(NV_Ith_S(ew,1)) + "]");
            }
            if (Ns) CHK(CVodeGetSens(mem, &tret, yS), "CVodeGetSens");
            record(k);
            if (!crossed && std::fabs(times[k] - d.tr) <= eps_t) restart();
        }
    } catch (const std::exception& e) { err = e.what(); }
    if (Ns) N_VDestroyVectorArray(yS, Ns);
    N_VDestroy(y); N_VDestroy(abstol); CVodeFree(&mem); SUNLinSolFree(LS); SUNMatDestroy(A); SUNContext_Free(&ctx);
    if (!err.empty()) throw std::runtime_error(err);
    return out;
}

static Opts make_opts(double rtol, double atol, long max_steps, bool analytic_sens, bool analytic_jac, double atol_theta,
                      int max_err_test_fails, double max_step, int max_order, bool current_weighted, bool sens_errcon) {
    Opts o; o.rtol = rtol; o.atol = atol; o.max_steps = max_steps; o.analytic_sens = analytic_sens; o.analytic_jac = analytic_jac;
    o.atol_theta = atol_theta; o.max_err_test_fails = max_err_test_fails; o.max_step = max_step; o.max_order = max_order;
    o.current_weighted = current_weighted; o.sens_errcon = sens_errcon; return o;
}

// Returns (I[n], theta[n], dI_dp[n, Ns], dtheta_dp[n, Ns]).
py::tuple simulate_sens(const std::vector<double>& times, const std::unordered_map<std::string, double>& params,
                        const std::vector<std::string>& sens_names, double rtol, double atol, long max_steps, bool analytic_sens,
                        bool analytic_jac, double atol_theta, int max_err_test_fails, double max_step, int max_order, bool current_weighted, bool sens_errcon) {
    Opts o = make_opts(rtol, atol, max_steps, analytic_sens, analytic_jac, atol_theta, max_err_test_fails, max_step, max_order, current_weighted, sens_errcon);
    Out r;
    { py::gil_scoped_release nogil; r = solve_core(times, params, sens_names, o); }
    const py::ssize_t nt = r.nt, Ns = r.Ns;
    py::array_t<double> I(nt), th(nt), SI(std::vector<py::ssize_t>{nt, Ns}), Sth(std::vector<py::ssize_t>{nt, Ns});
    std::copy(r.I.begin(), r.I.end(), I.mutable_data()); std::copy(r.th.begin(), r.th.end(), th.mutable_data());
    std::copy(r.SI.begin(), r.SI.end(), SI.mutable_data()); std::copy(r.Sth.begin(), r.Sth.end(), Sth.mutable_data());
    return py::make_tuple(I, th, SI, Sth);
}

// Solve many parameter sets (e.g. the nodes of a dispersion quadrature) on a thread pool. Returns (I[nb, nt], dI_dp[nb, nt, Ns]).
// A failure in any solve is reported (with its index) after all threads have finished.
py::tuple simulate_sens_batch(const std::vector<double>& times, const std::vector<std::unordered_map<std::string, double>>& param_sets,
                              const std::vector<std::string>& sens_names, int nthreads, double rtol, double atol, long max_steps,
                              bool analytic_sens, bool analytic_jac, double atol_theta, int max_err_test_fails, double max_step,
                              int max_order, bool current_weighted, bool sens_errcon) {
    Opts o = make_opts(rtol, atol, max_steps, analytic_sens, analytic_jac, atol_theta, max_err_test_fails, max_step, max_order, current_weighted, sens_errcon);
    const size_t nb = param_sets.size(), nt = times.size(), Ns = sens_names.size();
    py::array_t<double> I(std::vector<py::ssize_t>{(py::ssize_t)nb, (py::ssize_t)nt});
    py::array_t<double> SI(std::vector<py::ssize_t>{(py::ssize_t)nb, (py::ssize_t)nt, (py::ssize_t)Ns});
    double* pI = I.mutable_data(); double* pSI = SI.mutable_data();
    std::vector<std::string> errs(nb);
    {
        py::gil_scoped_release nogil;
        std::atomic<size_t> next(0);
        auto worker = [&]() {
            for (size_t b = next++; b < nb; b = next++) {
                try {
                    Out r = solve_core(times, param_sets[b], sens_names, o);
                    std::copy(r.I.begin(), r.I.end(), pI + b * nt);
                    std::copy(r.SI.begin(), r.SI.end(), pSI + b * nt * Ns);
                } catch (const std::exception& e) { errs[b] = e.what(); }
            }
        };
        int nth = std::max(1, std::min<int>(nthreads, (int)nb));
        std::vector<std::thread> pool;
        for (int i = 0; i < nth; ++i) pool.emplace_back(worker);
        for (auto& t : pool) t.join();
    }
    for (size_t b = 0; b < nb; ++b)
        if (!errs[b].empty()) throw std::runtime_error("node " + std::to_string(b) + ": " + errs[b]);
    return py::make_tuple(I, SI);
}

// Pointwise evaluation of f, df/dy and df/dp (all 9 parameters) at (t, I, theta). For unit tests.
static Data make_data(const std::unordered_map<std::string, double>& params) {
    Data d;
    for (int i = 0; i < NP; ++i) d.p[i] = (i >= iC1) ? opt(params, PNAMES[i], 0.0) : need(params, PNAMES[i]);
    d.E_start = need(params, "E_start"); d.E_reverse = need(params, "E_reverse");
    d.v = need(params, "v"); d.omega = need(params, "omega"); d.delta_E = need(params, "delta_E");
    d.phase = need(params, "phase"); d.cap_phase = opt(params, "cap_phase", d.phase); d.tr = need(params, "tr");
    for (int i = 0; i < NP; ++i) d.plist[i] = i;
    return d;
}
py::tuple eval_terms(double t, double I, double theta, const std::unordered_map<std::string, double>& params) {
    Data d = make_data(params);
    SUNContext ctx; SUNContext_Create(SENS_COMM_NULL, &ctx);
    N_Vector y = N_VNew_Serial(2, ctx), f = N_VNew_Serial(2, ctx);
    NV_Ith_S(y, 0) = I; NV_Ith_S(y, 1) = theta;
    rhs(t, y, f, &d);
    SUNMatrix J = SUNDenseMatrix(2, 2, ctx);
    jac(t, y, f, J, &d, nullptr, nullptr, nullptr);
    // df/dp via rhsS with S=0
    N_Vector* S = N_VCloneVectorArray(NP, y); N_Vector* Sd = N_VCloneVectorArray(NP, y);
    for (int i = 0; i < NP; ++i) N_VConst(0.0, S[i]);
    rhsS(NP, t, y, f, S, Sd, &d, nullptr, nullptr);
    py::array_t<double> fo(2), Jo(std::vector<py::ssize_t>{2, 2}), dfp(std::vector<py::ssize_t>{2, NP});
    for (int i = 0; i < 2; ++i) { fo.mutable_data()[i] = NV_Ith_S(f, i); for (int j = 0; j < 2; ++j) Jo.mutable_data()[i*2+j] = SM_ELEMENT_D(J, i, j); }
    for (int j = 0; j < NP; ++j) for (int i = 0; i < 2; ++i) dfp.mutable_data()[i*NP+j] = NV_Ith_S(Sd[j], i);
    N_VDestroyVectorArray(S, NP); N_VDestroyVectorArray(Sd, NP); N_VDestroy(y); N_VDestroy(f); SUNMatDestroy(J); SUNContext_Free(&ctx);
    return py::make_tuple(fo, Jo, dfp);
}

PYBIND11_MODULE(SurfaceODESensitivity, m) {
    m.doc() = "FTACV current and forward sensitivities dI/dp via CVODES";
    m.def("simulate_sens", &simulate_sens, py::arg("times"), py::arg("params"), py::arg("sens_names"),
          py::arg("rtol") = 1e-8, py::arg("atol") = 1e-8, py::arg("max_steps") = 1000000L, py::arg("analytic_sens") = true, py::arg("analytic_jac") = true, py::arg("atol_theta") = -1.0, py::arg("max_err_test_fails") = 50, py::arg("max_step") = 0.0, py::arg("max_order") = 0, py::arg("current_weighted") = false, py::arg("sens_errcon") = true);
    m.def("simulate_sens_batch", &simulate_sens_batch, py::arg("times"), py::arg("param_sets"), py::arg("sens_names"), py::arg("nthreads") = 4,
          py::arg("rtol") = 1e-8, py::arg("atol") = 1e-8, py::arg("max_steps") = 1000000L, py::arg("analytic_sens") = true, py::arg("analytic_jac") = true, py::arg("atol_theta") = -1.0, py::arg("max_err_test_fails") = 50, py::arg("max_step") = 0.0, py::arg("max_order") = 0, py::arg("current_weighted") = false, py::arg("sens_errcon") = true);
    m.def("eval_terms", &eval_terms, "f, df/dy, df/dp at a point (unit-test hook)");
    m.attr("PARAMETER_NAMES") = std::vector<std::string>(PNAMES, PNAMES + NP);
}
