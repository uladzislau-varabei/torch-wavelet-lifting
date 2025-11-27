from wavelets.bior_spline_33 import (
    fast_biorspline33_2d_op, fast_inv_biorspline33_2d_op, BIOR_SPLINE_33_KERNEL
)
from wavelets.bior_spline_35 import (
    fast_biorspline35_2d_op, fast_inv_biorspline35_2d_op, BIOR_SPLINE_35_KERNEL
)
from wavelets.bior_spline_37 import (
    fast_biorspline37_2d_op, fast_inv_biorspline37_2d_op, BIOR_SPLINE_37_KERNEL
)
from wavelets.bior_spline_39 import (
    fast_biorspline39_2d_op, fast_inv_biorspline39_2d_op, BIOR_SPLINE_39_KERNEL
)
from wavelets.bior_spline_48 import (
    fast_biorspline48_2d_op, fast_inv_biorspline48_2d_op, BIOR_SPLINE_48_KERNEL
)
from wavelets.rev_bior_spline_33 import (
    fast_revbiorspline33_2d_op, fast_inv_revbiorspline33_2d_op, REV_BIOR_SPLINE_33_KERNEL
)
from wavelets.rev_bior_spline_35 import (
    fast_revbiorspline35_2d_op, fast_inv_revbiorspline35_2d_op, REV_BIOR_SPLINE_35_KERNEL
)
from wavelets.rev_bior_spline_37 import (
    fast_revbiorspline37_2d_op, fast_inv_revbiorspline37_2d_op, REV_BIOR_SPLINE_37_KERNEL
)
from wavelets.rev_bior_spline_39 import (
    fast_revbiorspline39_2d_op, fast_inv_revbiorspline39_2d_op, REV_BIOR_SPLINE_39_KERNEL
)
from wavelets.rev_bior_spline_48 import (
    fast_revbiorspline48_2d_op, fast_inv_revbiorspline48_2d_op, REV_BIOR_SPLINE_48_KERNEL
)
from wavelets.cdf_53 import fast_cdf53_2d_op, fast_inv_cdf53_2d_op, CDF_53_KERNEL
from wavelets.cdf_97 import fast_cdf97_2d_op, fast_inv_cdf97_2d_op, CDF_97_KERNEL
from wavelets.haar import fast_haar_2d_op, fast_inv_haar_2d_op, HAAR_KERNEL
from wavelets.daub_4 import fast_daub4_2d_op, fast_inv_daub4_2d_op, DAUB4_KERNEL
from wavelets.coif_12 import fast_coif12_2d_op, fast_inv_coif12_2d_op, COIF12_KERNEL


WAVELETS_LIST = [
    ['CDF-9/7', fast_cdf97_2d_op, fast_inv_cdf97_2d_op, CDF_97_KERNEL],
    ['CDF-5/3', fast_cdf53_2d_op, fast_inv_cdf53_2d_op, CDF_53_KERNEL],
    ['Haar', fast_haar_2d_op, fast_inv_haar_2d_op, HAAR_KERNEL],
    ['Daubechies-4', fast_daub4_2d_op, fast_inv_daub4_2d_op, DAUB4_KERNEL],
    ['Coiflet-12', fast_coif12_2d_op, fast_inv_coif12_2d_op, COIF12_KERNEL],
    ['Bior_spline-3/3', fast_biorspline33_2d_op, fast_inv_biorspline33_2d_op, BIOR_SPLINE_33_KERNEL],
    ['Bior_spline-3/5', fast_biorspline35_2d_op, fast_inv_biorspline35_2d_op, BIOR_SPLINE_35_KERNEL],
    ['Bior_spline-3/7', fast_biorspline37_2d_op, fast_inv_biorspline37_2d_op, BIOR_SPLINE_37_KERNEL],
    ['Bior_spline-3/9', fast_biorspline39_2d_op, fast_inv_biorspline39_2d_op, BIOR_SPLINE_39_KERNEL],
    ['Bior_spline-4/8', fast_biorspline48_2d_op, fast_inv_biorspline48_2d_op, BIOR_SPLINE_48_KERNEL],
    ['Rev_bior_spline-3/3', fast_revbiorspline33_2d_op, fast_inv_revbiorspline33_2d_op, REV_BIOR_SPLINE_33_KERNEL],
    ['Rev_bior_spline-3/5', fast_revbiorspline35_2d_op, fast_inv_revbiorspline35_2d_op, REV_BIOR_SPLINE_35_KERNEL],
    ['Rev_bior_spline-3/7', fast_revbiorspline37_2d_op, fast_inv_revbiorspline37_2d_op, REV_BIOR_SPLINE_37_KERNEL],
    ['Rev_bior_spline-3/9', fast_revbiorspline39_2d_op, fast_inv_revbiorspline39_2d_op, REV_BIOR_SPLINE_39_KERNEL],
    ['Rev_bior_spline-4/8', fast_revbiorspline48_2d_op, fast_inv_revbiorspline48_2d_op, REV_BIOR_SPLINE_48_KERNEL],
]

FW_KEY = "forward_2d_op"
BW_KEY = "backward_2d_op"
KERNEL_KEY = "kernel"

WAVELETS_DICT = {
    w[0].lower(): {
        FW_KEY: w[1],
        BW_KEY: w[2],
        KERNEL_KEY: w[3]
    } for w in WAVELETS_LIST
}

WAVELETS_DICT_V2 = {w[0].lower(): [w[1], w[2], w[3]] for w in WAVELETS_LIST}
