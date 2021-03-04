#pragma once

#include <armadillo>
#include <set>
#include <exception>
#include <algorithm>

using namespace std;
using namespace arma;

class GMP
{
    public:
        typedef set<int> iset;

        GMP(void) {}
        GMP(iset Ka, iset La, iset Kb, iset Lb, iset Mb, iset Kc, iset Lc, iset Mc)
            : Ka(Ka), La(La), Kb(Kb), Lb(Lb), Mb(Mb), Kc(Kc), Lc(Lc), Mc(Mc)
        {
            //check parameters for validity
            if(Ka.count(1)==0) { throw runtime_error("Ka must include 1"); }
            if(Ka.size()==0) { throw runtime_error("Ka must have at least one element"); }
            if(La.size()==0) { throw runtime_error("La must have at least one element"); }

            if(*Ka.begin() < 1) { throw runtime_error("Ka elements must be >= 1"); }
            if(*La.begin() < 0) { throw runtime_error("La elements must be >= 0"); }

            if(Kb.size()*Lb.size()*Mb.size() > 0) {
                if(*Kb.begin() < 1) { throw runtime_error("Kb elements must be >= 1"); }
                if(*Lb.begin() < 0) { throw runtime_error("Lb elements must be >= 0"); }
                if(*Mb.begin() < 1) { throw runtime_error("Mb elements must be >= 1"); }
            }

            if(Kc.size()*Lc.size()*Mc.size() > 0) {
                if(*Kc.begin() < 1) { throw runtime_error("Kc elements must be >= 1"); }
                if(*Lc.begin() < 0) { throw runtime_error("Lc elements must be >= 0"); }
                if(*Mc.begin() < 1) { throw runtime_error("Mc elements must be >= 1"); }
            }
        }

        ~GMP() {}

        static int set_max(const iset in) {
            int m=0;
            if (!in.empty()) m = *in.rbegin();
            return m;
        }

        /* Compute maximum order of model */
        int order(void) { return max({set_max(Ka), set_max(Kb), set_max(Kc)}); }

        /* Compute number of samples in history (lookback) */
        int history(void) { return max({set_max(La), set_max(Lb)+set_max(Mb), set_max(Lc)}); }

        /* Compute number of samples in future (lookahead) */
        int future(void)  { return set_max(Mc); }

        /* Compute number of coefficients in the model */
        int num_coeffs(void) { return Ka.size()*La.size()
                                    + Kb.size()*Lb.size()*Mb.size()
                                    + Kc.size()*Lc.size()*Mc.size();
                             }

        //The contract!!!!!!
        //The user must supply a pointer with at least max(L_a,L_b+M_b) items
        //valid before the start of the vector, AND M_c items
        //after the end of the vector.
        template<class Ti, class To> Mat<std::complex<To>> gen_basis_matrix(const std::complex<Ti>* const in, size_t len);

        iset Ka, La;
        iset Kb, Lb, Mb;
        iset Kc, Lc, Mc;
};

template<class Ti, class To>
Mat<std::complex<To>> GMP::gen_basis_matrix(const std::complex<Ti>* const in, size_t len)
{
    const size_t nsamps = len+history()+future(); //length of the intermediate vectors

    Mat<std::complex<To>> X(len, num_coeffs()); //the output

    //input vector recast into an Armadillo column vector
    const Col<std::complex<Ti>> x(const_cast<std::complex<Ti>*>(in),  //pointer to memory
                                  nsamps, //length of col vector
                                  false, //don't copy it into new memory
                                  true); //strict preservation of vector size

    //to avoid having to calculate it twice (for MP and then GMP),
    //we calculate the power series of abs(x) here and store it.
    //xabsp[0] = abs(x)^0, xabsp[1] = abs(x)^1 ....
    Col<To> xabs = conv_to<Col<To>>::from(abs(x));
    Mat<To> xabsp(nsamps, order()+1);
    auto running_xabsp = Col<To>(nsamps, fill::ones);
    for (int i=0; i<=order(); i++) {
        xabsp.col(i) = running_xabsp.eval(); //late eval so the next line doesn't run unnecessarily on last iter
        running_xabsp = running_xabsp % xabs;
    }

    //generate MP terms (aligned signal and envelope)
    size_t colidx = 0;
    for (auto k : Ka) { //for each order
        auto xpow = (x % xabsp.col(k-1)).eval(); //x .* abs(x)^k
        for(auto l : La) { //for each delay
            X.col(colidx++) = xpow.rows(history()-l,history()-l+len-1); //output lagged x.^pow to X
        }
    }

    //generate GMP terms (signal and lagging/leading envelope)
    //envelope lagging terms
    for (auto k : Kb) {
        for(auto l : Lb) {
            auto x_lag = x.rows(history()-l,history()-l+len-1);
            for (auto m : Mb) {
                auto xabsp_lag = xabsp.col(k-1).rows(history()-l-m, history()-l-m+len-1);
                X.col(colidx++) = x_lag % xabsp_lag;
            }

        }
    }

    //envelope leading terms
    for (auto k : Kc) {
        for(auto l : Lc) {
            auto x_lag = x.rows(history()-l,history()-l+len-1);
            for (auto m : Mc) {
                auto xabsp_lead = xabsp.col(k-1).rows(history()-l+m, history()-l+m+len-1);
                X.col(colidx++) = x_lag % xabsp_lead;
            }
        }
    }

    return X;
}
