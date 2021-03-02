#pragma once

//The contract!!!!!!
//The user must supply a pointer with at least max(L_a,L_b+M_b) items
//valid before the start of the vector, AND M_b (should be M_c) items
//after the end of the vector.

#include <armadillo>

using namespace arma;

//TODO FIXME
//the "index array" idea in eq. 24 is useful. we should be able to specify
//a mask of Ka,La, Kb,Lb,Mb, Kc,Lc,Mc where we calculate only the relevant
//columns. this way, we can play with eliminating unused columns and try to
//get rid of low-value coefficients.
template<class Ti, class To>
Mat<std::complex<To>> gen_GMP_basis_matrix(const std::complex<Ti>* const in,
                                           int len,
                                           int K_a,
                                           int L_a,
                                           int K_b,
                                           int L_b,
                                           int M_b,
                                           int K_c,
                                           int L_c,
                                           int M_c)
{
    const size_t num_coeffs = K_a*L_a + K_b*M_b*L_b + K_c*M_c*L_c;
    const size_t history = std::max(L_a, L_b*M_b); //need to look back this far
    const size_t nsamps = len+history+M_c; //length of the intermediate vectors -- M_c in order to look ahead

    Mat<std::complex<To>> X(len, num_coeffs); //the output

    //remember: K_a is the order (i.e., up to 5th order polynomial)
    //          L_a is the memory depth (i.e., up to 4 samples memory)
    //          K_b is the GMP order just like K_a
    //          M_b is the GMP cross-lag depth
    //          L_b is the GMP memory depth just like L_a

    //input vector recast into an Armadillo column vector
    const Col<std::complex<Ti>> x(const_cast<std::complex<float>*>(in-history),  //pointer to memory
                                  nsamps, //length of col vector
                                  false, //don't copy it into new memory
                                  true); //strict preservation of vector size

//    x.print("x:");
//    std::cout << "history: " << history << std::endl;

    //to avoid having to calculate it twice (for MP and then GMP),
    //we calculate the power series of abs(x) here and store it.
    //xabsp[0] = abs(x)^0, xabsp[1] = abs(x)^1 ....
    Col<To> xabs = abs(x);
    size_t maxpow = std::max(K_a-1, K_b);
    Mat<To> xabsp(nsamps, maxpow+1);
    auto running_xabsp = Col<To>(nsamps, fill::ones);
    for (int i=0; i<=maxpow; i++) {
        xabsp.col(i) = running_xabsp.eval(); //late eval so the next line doesn't run unnecessarily on last iter
        running_xabsp = running_xabsp % xabs;
    }

    //generate MP terms
    for (int k=0; k<K_a; k++) { //for each order
        auto xpow = (x % xabsp.col(k)).eval(); //x .* abs(x)^k
        for(int l=0; l<L_a; l++) { //for each delay
            X.col(k*L_a+l) = xpow.rows(history-l,history-l+len-1); //output lagged x.^pow to X
        }
    }

    //now let's do lead/lag for GMP
    size_t colidx = K_a*L_a;
    //NOTE: the column order of X is in Kb,Lb,Mb order as per the paper, instead of Kb,Mb,Lb as per gr-dpd.
    //it really doesn't matter so long as this function is used by anything that interacts with coeffs.
    for (int k=1; k<=K_b; k++) { //why does this term go all the way up to K_b instead of K_b-1?
        for(int l=0; l<L_b; l++) {
            auto x_lag = x.rows(history-l,history-l+len-1);
            for (int m=1; m<=M_b; m++) {
                auto xabsp_lag = xabsp.col(k).rows(history-l-m, history-l-m+len-1);
                X.col(colidx) = x_lag % xabsp_lag;
                colidx++;
            }

            for (int m=1; m<=M_c; m++) {
                auto xabsp_lead = xabsp.col(k).rows(history-l+m, history-l+m+len-1);
                X.col(colidx) = x_lag % xabsp_lead;
                colidx++;
            }
        }
    }
    return X;
}
