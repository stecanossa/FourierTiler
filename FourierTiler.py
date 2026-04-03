"""
Fourier Tiler — Command-line version
Usage:  python FourierTiler.py Input.txt
        python FourierTiler.py               (looks for Input.txt in the working directory)

All parameters are read from the input file. Run with --template to write a
commented template input file and exit.
"""

import sys
import os
import random
import datetime
import math

import numpy as np
import numpy.fft as fft
from PIL import Image, ImageDraw, ImageOps, PngImagePlugin


# ---------------------------------------------------------------------------
# Colormap LUT data — exact matplotlib values, compressed
# ---------------------------------------------------------------------------
import zlib, base64
_ALL_CMAPS = [
    "inferno",
    "viridis",
    "magma",
    "OrRd",
    "PuRd",
    "RdPu",
    "BuPu",
    "GnBu",
    "PuBu",
    "YlGnBu",
    "PuBuGn",
    "BuGn",
    "YlGn",
    "gist_yarg",
    "gist_gray",
    "bone",
    "pink",
    "hot",
    "afmhot",
    "gist_heat",
    "PiYG",
    "PRGn",
    "BrBG",
    "PuOr",
    "RdGy",
    "RdBu",
    "RdYlBu",
    "RdYlGn",
    "Spectral",
    "coolwarm",
    "seismic",
    "twilight",
    "twilight_shifted",
    "hsv",
    "prism",
    "ocean",
    "gist_earth",
    "terrain",
    "gist_stern",
    "gnuplot",
    "gnuplot2",
    "CMRmap",
    "cubehelix",
    "brg",
    "gist_rainbow",
    "rainbow",
    "jet",
    "turbo",
    "nipy_spectral",
    "gist_ncar",
]
_N_CMAPS = len(_ALL_CMAPS)
_LUT_B64 = (
    "eNq8u3VUVN8b9r0HkbC7BQmxuxVRkRQBQUGQ7kaku5uhu7u7u7u7uzuGYciZ2e9xfL7P+q31/P/e"
    "6/qDpcOZffa+rs999rAHgEMAkANAgQKUKBQ1CnWUDHWCjOzUIbIz5IfOkR+6cJj8EgX5VcrD16ko"
    "aKkp6I5QMhylZDpOdfsE1d2T1A9OUT86c+Tp2SPPzx19eeHom4tH310+ynLl6Mdrxz5fP8ZOe4zz"
    "xrEv9Ee/Mh7lYzr67dZRwTtHf9w7IvzgiMjDI78eHxF/Si3xnFrqBbX0K2qZN9Sy76jlmakVWKgV"
    "P1IrfqJW+nxEmf2ICscRFa6jql+OqvIcU+M9psZ/XP3bcXXBE+rfT2gIndT4eVJD5JTGr9MaYqc1"
    "Jc5oSp3RlD6rKXtOU+6cpsJ5TcXzmsoXNFUvaqpd1NS4pKl5SVPrsqb2ZQ2dKxp6VzX0r2oYXtMw"
    "vqZhcl3d7Lq6OY26Ja2aFa2azQ01uxuq9nSqjnSqTvQqLvQqrgwqbozKHozKnjeVvG8q+TIp+jEp"
    "BtxSCLwtH3xbPuSOXNgduYi7spH3ZKLvycTcl467LxX/QCrxoWTyQ4mUR+Jpj8XSn4hlPvmV9VQ0"
    "55lI7rOf+c+FC14IF70UKn71o/TV97LXguVvBCrffqt6x1/DzFf7nree5WsDC0/jhy/NH7lbPnG1"
    "snK2s3F0sLN1cnzu4mTt4frUy/2x78uHfh6WQd73Q3zvhvnfjnx7Myr4euz7q/EfLyeEn0/9fDYt"
    "8nTm15NZscdz4g/nJB/MS91fkLm3KHt3Sf72ssKtFSWmVeWbq6oMa+r06xp0G79pN7RoMNrXN3Wv"
    "bepdwRpcxhpd2jK+gDM9jzM/t215Ztvq9I7NqR27E7sOx3cdj+06H91DU++5Ue15UO17Uu57U+z7"
    "Uuz7UxwEUhwEUxyEUhyEUR5EUB5EUR3EHDmIO3qQcPwg6cRByqmDtDMHGecOMi8cZF86yL1ykH/t"
    "oJBmv5huv5Rhv/zmfuXtveq7e7X39+of7TY+2W1+vtv6aqf97U4n83b3h+1eVlw/O26Qa2uYZ2uE"
    "HzsmiJ0Qxk6Jbs5IbM5JYxbkMYvKmGV1zKoWZk0Xs2GE2TTbxFptbtlht522dlxxu57b+747B4F7"
    "+NB9QgSeGMOG+sFGJsR26Cf7YRF2il/sVGIc1OIcRyU5jklxnJDhPCnLeVqO84wC5zlFzvNKnBdU"
    "OC+pcl1W47qiznVNk+v6by4aLa4bf7jotLnodbgYdDlv6nEy6XPeMuC8bch514jznjHnfROOB6Yc"
    "j0w5HptxPDFnf2rB/tyC/YUl+0srtldWbG+sP7+1+fzO5jOzLet7W1YWO9aPdp8+2X9itf/42eEj"
    "m8MHdocPnI4sXI4s3E7vvzi953F6/9WJmdeZmc/53TfndwLObwVd3n53efPD5Y2Qy2thl9c/0a9E"
    "0K9E0S9/oV+KoV+Io19IoJ9Luj6Xcn0m7fpMxvWprOtTOdcn8q5PFFyfKLo+VnJ9rOz6SMX1karr"
    "QzW3h+puDzXcHmi6Pfjtdl/L7f4ft/vabvd03O7put3Vc7ur73bXwO2OodsdI7fbxiSZuN0ydbtl"
    "5nbL3I3Jwo3J0o3Jyu2mtdtNG1dGW1dGO1dGe1cGB1cGR1cGJ1d6RM6u9C6udGhXOlc0nRv6hjv6"
    "hgf6hiea1gtN6+1C6+NC6+tC6+dC4+9ME+BME0hSkBNNsBNNiBNNqCNNmCNNuCNNhANtpANtlD1t"
    "tP2NGLsbsXZ0cXZ08bb08bYMCTYMidaMSdY3k62YUqxupVreTrO4k25xN8P8XqbZ/Uyzh1mmj7JN"
    "HucYP801fpZn9CLf8GWBwetC/TeF+u+K9N4X67KU6Hws1WYt+8NWpsVe/puzQpO7UoOnSp23So2/"
    "WkWgRvl7rZJQneLPOgXRenmxBlnJRhnpRmnZJin5ZgmlFnGVFjG1VlGNNhGtNmHtdiHdjh8GHYJG"
    "nQKmXfzmXXxW3V9tunnse7gdezldejlc+9g8+j579X/y7f/gP8ASNMAcOvg2fPB11NCr2KEX8cPP"
    "koafpo48Th95mDV6P3f0Xv7onaKxW6VjTOXjjFXj9LXjdPUTtE0TNK0T19snr3VNXu2ZvNI/dXlo"
    "6tLI1KXx6UuT05dmpi/Nz1xenLm8MnNlbfYqZvYadpZme452d45uf46eMM8I/l/+kx0nIzt56NBp"
    "8kNnycnPHya/SHH4MuXhq1QU16kpaY9Q0h+lYjxGzXT8yO0TR+6ePHr/1LFHp489OXP82bnjL86f"
    "eH3hxNuLJ5kvnfxw5dSnq6fYrp3ioDnFTXua58ZpXvrT3xhOC948LcR0+uft06J3T4vfOy354JT0"
    "o1Oyj08pPD2l9PyU8ouTqq9Oqr85qfnu5G/mk39YTmp/PKnDelL380k99pP6nCcNuE4ZfjllxHPK"
    "iPe0Mf9pk29nTATPmH4/ayp01uznOTOR8+a/zpuLXTCXuGguedFC+pKFzCULucsWClcsFa9YKl+1"
    "VLlmqXbNUv26pSaN1W8aqz+0Vjq0Vro3rPTprAzorIzorYzprUwZrMwZrSwYraxuWlnftLJlsrJj"
    "snK4ZeV028r5thX6jqXrHUv3u5aedy297ln63LP0vW/pf98i8IFF0AOLkIcWYQ/Nwx+ZRz4yj35s"
    "HvPYLO6JWfwTs8SnpsnPTFOemaQ9N8l4bpL5wjj7hXHuS6O8l0YFrwwLXxkWvzYofWNQ9ka/4q1e"
    "1Vu96ne6tcy6dcw6De+1m1i0m1n+tH7Qavuo1fHpdyerZjerRs9njT429X52tUEO1SFOlREulVFu"
    "5bEvShM8ipNfFad4FWb45Ge/yc0JyM4Lyi78kFkSkl4Wll4RkVoVlVoTk1yTkFiXlNiQlsDIiG/K"
    "iWMVxLFKYlvKYjhVMZy62Lam2LaW2M4fsR0d8V098V0D8T0j8T0TiT0ziX0LyX1LyX1rqQNb6QN7"
    "6QNHmQNnWTxaDu8mh/eQx3sp4L0VCb5KBH9lQqAKIViVEKpGCFcnRGoSon4TYrSIcX+ICTrEJF1i"
    "ih4xTZ+YbkjMNCJmmxBzTYn5ZsRCC2KxJbHEmlhmQ6ywI1bZE2sciXVOxHoXYqMrsdmN0OpBaPci"
    "dHoTunwJPf6EvgDCQBBhKIQwEkoYDSeMRxImo/HTsfjZOPxcAn4hCb+Ugl9Jw69l4Dcy8ZhsPDYX"
    "j8s/2Ck82Cs+2C89wJfDnRXi9hJxe5GIWyDi5olbc8StWSJ2hoidIm5OEjcniJhxImaMuDFK3Bgh"
    "rg8T1weJ6wPEtX7iWh9xtZe42kNc6SaudBGXO4jL7cSlNuJSK3GxhbjYTFxoIi40EObrCfN1hLla"
    "wlw1YbaKMFNJmCknTJcRpkoJU8WEySLCZCFhooAwnkcYzyWM5RBGswijmYSRDMJIGmE4lTCUQhhK"
    "IgwmEgYSCANxhP5YQl8MoS+K0BtJ6I0g9IQRukMJ3SGErhBCZzChI4jQHkhoDyC0BRBa/QktfoRm"
    "X0KzD6HJh9DoTWjwIjR4Euo9CHXuhFp3Qo0bocaVUI0mVLkQKl0IFc6ECidCuSOhzIFQav9XJXaE"
    "Ylt8kQ2+0Bqfb4nPs8DnmOOzTfGZJvgMY3yaIT7VAJ+ij0/SxSfq4OP/4OO08DG/8dEa+Eh1fIQa"
    "PlwFH6qMD1HCByngA+Xx/nJ4Pxm8jzTeWwrvJYH3EMe7i+FdRQ/QIgcuIvtOP/cdf+47CO/ZC+/Z"
    "Cu3aCO1a/9i1+r5j8X3HXHDbTHDbVGDbWABn9A1n+G3LgH9Lj39Llw+rw4fV5t3U4t38/XVTkwej"
    "wYNR+7Kh+mVDhXtdiXtdkWtdgWtNjnNVln1Vmm1F8vOyBOuS2Mcl0Q+LIiwLwszzP97Nf387J/Bm"
    "9turGb6X019fTPM8m+J+Osn5ZILj8QTbw3HWB2Of7o9+uDv6/s4I8+3ht7eGXt8cesU4+IJh4Bld"
    "/9Mb/Y9p+x7S9D643n3vWtfdq523r3Tcutx+81Ib48VW+gvNdOebaM810pxtuHam/urpusunai+d"
    "rLlwour88cqzxyrOHC0/daTsJHXpcaqSY5TFRygKqQ8XUJLnU5DnkR/KPUSWg0JlA5AJQDoAaQCk"
    "AJAMQCIA8QDEARADQBQAkQCEAxAKQAgAQQAEAuAPgC8APgB4AeABgDsArgCgAXAGwBEABwDsALAB"
    "wBoASwAsANjZ2tvG7uI2/2oLs4MIu7GDXd/eRLS2jVnDYVZxGyuIttaX/2ptCbu2iF1FtIBdWdhc"
    "md9cnkOEWZr9q8WZjcXpjQVEU+vziCbX5ybW5sbXZsdWZ0dXZoZXpoeWpweXpgaWpvoWJ3sXJnoW"
    "JrrnxzvnxzvmxtpnR9tmR1tmRpqnR5qmhxunhusnh+omB2snBqvHB6rGByrH+itG+8pG+0pHekuG"
    "e4uHewqHugsGu/MHu/MGunIHunL6u7L6OjP7OjN6O9N7O9J6OlK7O5K725O62hO72hM62+I72uI6"
    "2mLaW6PbW6PaWiNbWyJaW8JbmsNamkOam4ObmoKamgIbmwIaG/0bGn3rG33qG73rGj3rmjxqm91r"
    "WlyrW9DVrc5VbU6VbQ4V7fYVHbblHTZlnVZlXZalXeYl3WbFPSbFPcZFvYaFfQaF/XoF/br5Azp5"
    "g3/yBrVyhzRzhjWyh9WyR1SzRpUzR5UyxxQyxuXTx2XTxqVTxyWTJ8STJn4lTogkTAjHTwrFTn6P"
    "mRSInuSPmuKNnOIJn+IOm+IMnWYPmWYLnmYNnP4YMMPiP8PsN/PWd+a19+xLr9nnnrPPPGafuM89"
    "cpt7gJ675zJ3x3nutuPsLYeZW3YzTLbTN62nGC0nGS0mGcwm6E3G6Y3H6AxHbxiM0uqN0OoM02gP"
    "XdcavK45eE1j4Kpa/xWVvivKfZcVey8p9FyU674o03VBuuu8ZOd58Y5zYu1nRdvOiLSdEW45LdR8"
    "+kfjqR/1J7/Xnvxec0Kw6oRAxXGB8uPfSo99Kz7KX3SUr+AIX94R3lxq3mzqr5lUX9MpedIov6RQ"
    "fEmi4E48zB1/mCuWnDOGnDPqEEcEGUcYGXsoii0YxRYIPgeAz36A1Rd89AYfvACLB3jvBphdwTsX"
    "8NYZvHYErxzASzvwwhY8twbPrMBTC/DEHDwyBQ+NwQMjcN8A3NMHd3XBHR1w+w9g0gI3NQGjBmBQ"
    "A/SqgE4Z3FACtHAHS9zeJOIwxK2Nv8KuEzfXiBhEq8SNFeI6omXi2hJxdYm4skhcWSAszxOWEM0R"
    "FmcJC7OE+RnC/DRhbpowO0WYmfyr6QnC1DhhaowwOUaYGMWPj+DHh/Fjw/jRIfzIIH54AD/Ujx/s"
    "ww/24gd68P3d+L4ufG8nvgdRB767Hd/Vhu9sxXe04Nub8W1N+LZGfGsDvqUe31yPb6rDN9biG2vw"
    "DdX4+ip8XSW+tuKgpvyguvygqvygsuygvOygrOygtPSgpPSguPSgsOSgoOQgv+Qgr/ggt/ggp/gg"
    "q2g/s2g/o2g/vXA/rXA/pXA/uWA/qWA/MX8/IX8/Ln8/Nm8/Jm8/Om8/Knc/Inc/PHcvLGcvJGcv"
    "KHsvMGvPP2vPN3PPJ3PPK2PPI33PLX3PNW3XJXXXKXXXIWXXPnnXNnnXOmnXKmnXInHXLGHXJGHH"
    "OH7HMG5HP25HN3ZHJ3bnT8zO7+gdzegd9agd1cgd5chtpQicQsSWXMSWbDhWOnxTKhwjEbYhFrbx"
    "K2xdJGxNOHRVKHTle+iyQMjyt5AlvpBF3uAFnuB57uB5rqA5jqBZtqCZz4HTn5A4BE6xBE4yB0y8"
    "Cxh/EzD2yn/spf/oc/+RZ35DT3wHH/n0P/Tpu+/dc9er+45n5y3PDiaPNkb3Vga3ZjrXJlrXBhp0"
    "/TWX2qvONZedqi86VV5wrDjnUHbWvvS0ffFJu6ITtgXHbPKPWOdSW+dQWmVRWGaSW6STmaehzFNQ"
    "Zskos0SUWTzKLA5lGoMyjUKZRqJMw1GmYSiTEJRJEMokEGXijzL2RRn7oIy9UMaeKGN3lJErygiN"
    "MnJGGTmijBxQhnYoQxuUoTXK0BJlYIEyMEMZmKAMjFEGhkDfAOjrAT1doKcN9LSA7m+gqwF01IGO"
    "KtBWAdpKQFsR/JEHf+SAlgzQkgZakuC3OPgtBjRFgaYI0BQGGkJA4ztQFwTq34A6P1DjBWpfgeoX"
    "oMoNVHfwhO0DPG4fv4Vo7wC7d7C5e4BBtLO/sbO/vo1obw23t4rbW9lCtLuM3V1CtLmzuLmzgNmZ"
    "R7SxPbexPbu+PbOOm1nDTa/hplYRbU2ubE2sbI0vY8eWsaNL2JHFzeGFzaF5zOA8ZmBuo392o29m"
    "vXdmvWd6vXtqrWtyrXNitWNitX18pW1spXV0pWV0uXlkuWl4qXFoqWFwsX5wsXZgoaZ/obpvoap3"
    "vrJ3vqJnrrx7rqxrtrRrtqRztrhjpqh9prB9pqBtOr91Oq91KrdlKqd5KrtpMqtpMrNxMqNhIr1+"
    "Iq1+PLVuPKV2PLl2LKlmLLF6LL5qNK5qNLZyJKZiJLpiJKp8OLJsOKJ0KLx0KKxkKLR4MLRoMKRw"
    "ICS/PzivPyi3LyinLzC7NzCrJyCjJyC92z+tyz+1yy+l0zepwzexwyeh3Se+3TuuzTum1Su61TOq"
    "xTOy2SOi2SOsyT20yT2k0S24wTWowTWgHu1fh/arQ/vWor1r0V41aM9qtHs12q0K7VqJdqlEO1eg"
    "ncrRDuVo+zK0XSnaphRtXYK2KkZbFKPNi9BmhWiTQrRxAdowH22Qj9bPQ+vmonVy0do5aK1s9O9s"
    "tGYWWj0TrZbpoprhopzuopTuopjmIp/mIpfqIpviIp3iLJXsLJnkLJHkLJbo/CvRWTTB+We8s3C8"
    "k1Cc0/dYJ8FYJ4EYJ/4YJ75oJ94oJ54oxy+RjtwRjpwRjhzhjuzhjp/DHFlDHT+FOnwIcWAJdmAO"
    "cngXaP8mwP6Vv/1LP/vnPvbPvO2eeNk99rR76GF33932npvtHVfb22hbJhfbm842DE429I42Nxxs"
    "aOxtrttZX7W1vmJjfcna+qKV1XlLq7MWVmfMrU6ZWZ00tThhbHbCyPS4gfFxfaNjugbHtPWO/tE9"
    "+lv7iKbWEfXf1Goa1CpqVMqqVIrKVApKlHIKlLJyFNKyFJLShyUkD4tJkP8SIxf5deinyCEhYbIf"
    "QmSC31ECAij+b4CPD3zlBTw7+I3tg3XcwdoWov1V7P7K5v4KZn8Zs7e0sbe0vre4vruwtruwuju/"
    "sju3sjO3vDO7tDOztD2zuD29sD01jwg3OYebmMVNzGyNz2yNTW+NTW2NTmFHJrEjE9jh8c0hRGOb"
    "g2ObA6OYgRFM/wimbxjTN7TRO7TRM/hX3QPrXf3rnYj61jt619oR9ay19ay1dq+2dq22dK02d642"
    "d6w0daw0tq80tCFarm9drkPUslzbvFTTtFTTuFTdsFhdv1hVt1BVs1BZPV9ZNV9ROVdRMVdePlde"
    "NltWOltWPFNaNFNaOF1SMF2SP1WcN1WcO1WUM1mUNVmYOVGYMVGQPl6QNp6fOp6fMpaXNJaXOJqb"
    "MJobP5IbN5IbM5wXPZwXOZQfMZQfPlgQOlgQMlAQNFAY2F/o31fk11fk21vs3Vvs1VPi0VPi3l3q"
    "1l2K7ipz6Spz6ix37Cx36Kiw66iwba+wbqu0aqu0aK0yb60ya6k2aak2bq4xbKoxaKrVb6zVbajT"
    "qa/7U1evVVf/u7ZBo6ZBvbpBtapRpapRqbJJsaJJobxZrrxZtqxFurRFqqRVsrhVvLhNrKhNtLBd"
    "pKD9Z0GHUH7Hj7wOwdxOgZxO/pwuvuwu3qyur5mdPOmdX9I6uFLbOVPaOZLa2BNb2RJaP8e3sMY1"
    "f4pp/hjd9CGq6X1kI3N4w7uwhreh9W9C6l4H1b0KrH0ZUPPCv+a5X/VTn6on3lWPvSofeVY8dK94"
    "4FZ+37XsHrrsrkvpHefSW44lTA7FN+2LGe2KGGyL6G0K6awLb1gW0Frk05jnXzfLu2qad8Uk97Jx"
    "7iWjnIsG2Rf0s8/rZZ3TzTqrk3lGO/PUn4yTWhknNNOPa6QdU087qpZ6RDWVWiWFSjmFSjGJSiGR"
    "Sj6BSi6OSiaWSjqGSiqKSjKSSjyCSiyM6lcolWgw1c8gKuFAKiF/qh9+VIK+VALeVN+8qPg9qXjd"
    "qb66UfG4Un1xoeJypuJ0ouJwoGK3p/psR8VqQ/XJmuqjFdw5IG7vE3B/hTSAfz1gH9Hm7h4izO4u"
    "Zmdn46+21xFt49a2txCt4rCIVnCbKzjM8haijSVE2PXFv7uDtQVkd7C5Ov9vd4D5tztAtgYLMxvz"
    "0xtzU+uzU+szk2vTE381Nb46ObY6Mbo6ProyNrIyOrw8gmhoeXhwaWhwaXBgaaB/sb9/sa9vobd3"
    "obdnoadnvrt7vqtrvrNrrqNzrr1jtq1jtrV9trltpqltprF1pr51uq5lurZ5urp5qqppqrJxqrxx"
    "sqxhsrRhsrh+oqhuorBuIr92PK9mPLdmPKd6LKt6LLNqNKNyNK1yNLViJKV8JKl8JLFsOKFsOK50"
    "OLZkKLpkKKp4KKJoMLxoMBRpBIUDQQUDAfn9/vn9vnn9Prl9Xrl9njm97tm9rtm96Kwe56wep8we"
    "h4xu+4xu2/Qu67Quq7Qui9RO89RO05ROk+QOo+QOg6R2/cR23cR2nYS2P/Ftv+NbNeJa1WJbVGJa"
    "lKKbFaKb5aKaZSKbpCKaJMIbxcIbRcMafoY2CIXUfw+pFwiu+xZUxxdY+zWg9ktADZd/DYdfNZtv"
    "NatP9UefKhbvKmavyreela89K156VLxwL3/mVv7EtewxuuyhS+kD59J7TiV3HUtuOxTfsi++aVfE"
    "+J9daa0KrlsWXLPIv2Kef9ks76Jp3gWT3HPGuWeNck4b5pwyyD6hn31cL+uobha1TiaVdibFn4zD"
    "Whnkv9PJNdPINVPINZLJ1ZPI1RPJ1RLIVePJVeLIVWLJlWPIlaLJlaLIFSPIFcLJFcLI5UPJ5ULI"
    "5YLJZYPIZQLJpQPIpf3JpXzJJX3IJb3JJbzIxT3JxT3IxdzJf7mR/3I9JOpySMTpkLDjISH7Qz9s"
    "D323OSRodUjA4tA380P8pof4jA/xGh36anCIR+/QF91D3NpkXFpkHL/J2DXI2NTIPquSsSqTfVIk"
    "+6hA9kGOjEWG7L00GbMk2TtxsrdiZG9EyV6LQDhIJPYTiH0EYi+e2I0ndB0QOg8IHfuEtj1C6x6+"
    "ZRffvItv3ME37BzUbx/U4Q5qcAfVWwdVW/uV2P1y7H7Z5n7p5l4JZq9oY69wY69gfS9/fTd3bTdn"
    "bTd7dTdrZSdzeSdzaSdzcTtzYTtzDpc5i8uc2cqc3sqcwmZOYDPHN7PGNrNGMVkjmKyhjazBjayB"
    "9az+9ay+9azetazutayu1ezO1eyOlez2lezW5eyW5ezmpeympezGxez6xey6hZya+ZzK+ZzyuZzS"
    "2ZzimdyC6dy8qdycydysibz08bzU8bzkvzDPixvJjxnOj0IYPpgfOlAQPFAQ2F/g31fg01vg1VPo"
    "0V3o2lXo0lno1Fno0FFk215k3VZk2Vpk3lJs2lJs3Fxs2FSs31ii21ii3VCiVV+qWVeqXleqWluq"
    "XFOmWFMmX10mV1UuU1kuVVkuUVEuVl4hWl7xs6xCqLTye0mlQEklf3Elb1EVT2EVd2EVZ0E1e371"
    "5/zqT3nVH3OrP+RUs2RXv8+sfpdR/Ta9+k1a9evU6lcpVS+Tql4kVj1PqHoaX/UkrupxbNWjmKqH"
    "UZUPIivvR1TeDa+8E1Z5O7TyVnAlU1DlzcAKxoAKBv8KOr+KGz4VtN4VNF4V1z0rrnuUX3cru44u"
    "u+5Set2p5LpjMY19MY1tEY1NIY1VAY1lAcJnGtM8WpNcWqMcWsMcWv1sWt0sWp3MG38yb2hl3NBM"
    "v6GedkMt7YZKKp1yCp1iMp18Mp1cEp1MIp10Ap1kAr1EPL1YHP2vOHrRWPqfsfTCMfRC0fTfo+kF"
    "o+gFIhn4Ixn4Ihh4Ixh4whm+hDFwhzFwhjJwhDCwhzB8DmZgDWL8FMT4MZCRJZDxfQAjsz/jW3/G"
    "N36Mr30ZX/oyvvBhfO7D+MyL4akH/RM3uifoG4+daR850jxyuP7Q7toDm2v3ra7et7hyz+zyXZNL"
    "d4wu3jG8cFv//C3dc7e0zzJpnbmpeYZR/TSj6ikG5ZP0SifoFY7TyR27IXOUVuoIrQQ1jRjVdVGq"
    "6yL/w/+/8McjGwDsLgL/fQwC/529jZ3d9Z2d9e2dte3tVUQ43ApuaxnRFnZpC7u4tbmIxSxgMfPY"
    "jfnNjbnN9dnNtVnM2gxmdXoD0crUxvLkOqKlifXFifWF8bX58bW5sb8fB82Mrk4jGlmZGlmZHF6Z"
    "GF4eH1oeG1weHVwaGfir4f6lof7Fwb7Fgb7F/t6FPhL8e3vme7r/8X+us/P/P/5HFg/9hX/RYHDh"
    "QGDhgH/BgG9+v3d+v2dev3tun2tun0tOr1N2r0N2r11Wj01Wj1Vmj0VGt1lGt0l6l1Fal0Fal15q"
    "p05q55+Uzt/JHRrJHWpJ7SqJ7UqJ7YoJbfLxbXLxrTJxrdKxLZIxLRLRzWLRzaJRzSKRTcIRTULh"
    "jd/DGwXDGr6FNvCH1POG1H8NrvsSVMcdWMsZUMseUMPmX8PqV/3Jt/qDTzWLTxWzd9U7r8o3//H/"
    "uXvZM/fSJ27Fj90KH7oW3HfNu4fOuYPOvuWSyeSSzuiSxuCcQuecROuUQOMUf80x9opj9GWHqIsO"
    "Eeftw87Zh56xCz5tF3jS1v+4rd8xG58jNl5UNp6U1u6HrV3JrdBkVs4oS0eUhQPKwg5lbosys0aZ"
    "WaFMLVAm5igTU5SxCcrICGVkgDLURxnoogx0UPp/UHpaKD1NlK4GSkcNpaOK0lZGaSuh/iigtORQ"
    "WrKo39IoTSmUpgRKQxyl/gulLopSFUGpCKOUhVBKP1CK31Hygig5AZTsN5QMP0qaDyXFi5LgQYl/"
    "QYlxo35xoUQ5UT85UMLsKCE21I/PqO+sKIGPqG8fUPwsKL73KF5mFM871Je3KO43KK7XKM5XKI6X"
    "/2f/S9LWwd9dMJakzf0DDKK9gw2S1hHtHqztHqzu7iNaQbSzv0zSEqLt/cXtvQWS5hHh9uZImsXt"
    "zWztIpre2pna2pnEbk9gcYjGN7fGNrdGN7Gjm5sjmM1hDGYIgxnEbAxurA9srPdvrPWtI1rtXV/t"
    "WV/pXlvu+qulztWljtXF9tWFNkQr860r8y0rc83Ls03LMw1L03WL07WLU9ULk1ULExXz42VzY6Vz"
    "o8WzI0UzwwUzQ3nTg7nTg9lTA1mT/RmTfWkTvanjPcnj3UljXQljnXGjHbEj7dEj7VHDbRFDrWFD"
    "LaGDzcGDTUEDjQH9DX799b59dd69tV491R49VW7dla5dFS5d5c6dZY4dJfYdxXbtRTZthdZtBZat"
    "+eYtuWYtOSbN2cZNWYaNmfqNGXoN6Tr1qdr1KVp1yZq1SRq1iWo1CarVccrVsYpVMQqV0fKVkXIV"
    "ETLlYdLloVJlIZKlQRKlgeIl/r9K/ESLfUSKvH8WeQoXeggVuP8ocBXMRwvkO3/Lc+LPdeDLtefN"
    "seXJtvmSbc2dZcmVZcGZacaRYcqWYfw53Yg1zeBTmv7HVL0PKTosyX/eJ2kxJ2i+i9d4G6f2Jlb1"
    "dYzyq2ilV1EKLyPlX0TIPg+XeRYq9TRE4kmw+OOgX48CRR8G/HzgL3zf78c93+93fQTueH277cl3"
    "y4P3ljsPkxv3TVcuRjQHgws7vTMbndNnWkdWGgfW6/afrtl9vGr74YoNy2VrlktW7y9aMl+weHfe"
    "/O1Z87dnzN6cNn19yuTVSeNXJ4xeHjd8cczg+VH9Z0f0nlHpPqXUeUKh/fjwn8fkWo8O/X5I9vsB"
    "SvM+0LgP1O4B1XtA5S5QvgMU7wCF20D+NpC9BWRuAWkmIMUEJG4C8ZtAjBGIMgIRBvCTAQjTgx90"
    "4DsdELwBvt0A/LSAjxbw0gAeGvDlOuC+DjivAY5rgP0qYLsK4SyROE0kThGJEwTiOIE4RiCOEIjD"
    "eOIgnjiAJ/bjCb0HhJ4DQvf/fS7aJ7TuE1r2CU17hMY9QsMevm4PX7uLr97FV+3iK3fx5Tv4sh18"
    "yQ6+eAdftH1QgPur/K39POx+3uZ+LmYvZ2MvZ303e203678HpIylnfTF7fSF7bR5XOocLnUWlzK7"
    "lTyzlTy9lTSFTZzEJk5sJoxvxo9txo1i4kYwscOYmKGNmMGN6IH1qP71qN61yJ61iO7ViM6V8I6V"
    "sLblsNbl0JalkKbFkMbF4IaFoLqFoNr5wOq5gKq5gMpZ//JZv7IZv9Jp3+Jp36Ipn4Ip7/xJ77wJ"
    "r5wJz+xxz6xxj4wx9/RR99RRt5QR16QR18RhdPyQS+ygS8ygc9SAU0S/U3ifY2ifQ3CvQ1CPfUCP"
    "nV+3nW+XrXenjWenjUeHtVu7FbrdyqXN0qnVwqHFwr7F3LbZzLrJzKrR1KLRxKzBxLTe2LjeyLDO"
    "yKDWUK/GQKda/0+1nlaVnmalrnqFjlq5tkrZH6VSLYXS3/IlmrLFmtJFGpKF6hIFamIFqqL5KiJ5"
    "ysK5Sj9yFAWzFQWyFPiz5Hkz5XgyZL+ky3ClSXOkSrGnSn1OkfyULPExSfxDohhL/K/3cb+YY0Xf"
    "xYi8jfr5JlL4dYTQq3ChV2E/XoZ8fxEs+DxI4FmgwFP/b0/8+B/78j3y4X3o9fWB59f7Hjz33L/c"
    "deO+g+a67cJ1y5nzlhMHkwP7TXs2Rjs2BtvP9DasdNastFafaCw/Xbf4dM3841Xzj1fMPl42/XDJ"
    "5MNFY5YLRiznDVnOGrw/Y/D+tP77U3rMJ3WZT+i8O6797tifd0e13h7Rekv1+y2l5hsKjTeH1V+T"
    "q70+pPqaTOUVSvkVQKT0Eii8BPIvgNwLIPscyDwH0s+A1DMg+QxIPAXiT8GvJ0D0CRB5DH4+BsKP"
    "gdAj8OMR+P4QCD4EAg8A/wPAdx/w3gdf7wOee+DLPcB9F3DdBZx3AMcdwH4HQkgkEgkEAh6PPzg4"
    "2N/f39vb293d3dnZ2d7exuFwW1tbWCx2c3MTg8FsbGysr6+vra2trq6urKwsLy8vLS0tLi4uLCzM"
    "z8/Pzc3Nzs7OzMxMT09PTU1NTk5OTEyMj4+PjY2Njo6OjIwMDw8PDQ0NDg4ODAz09/f39fX19vZ2"
    "k6qrq6uzs7Ojo6O9vb2tra21tbWlpaW5ubmpqamxsbGhoaG+vr6urq62traaVFVVVZWVlRUVFeXl"
    "5WVlZaWlpSUlJcXFxUVFRYWFhQUFBfn5+Xl5ebm5udmkysrKyszMzMjISE9PT0tLS01NTUlJSU5O"
    "TkpKSkxMTEhIiI+Pj4uLi42NjSZVVFRUZGRkREREeHh4WFhYaGhoSEhIcHBwUFBQYGBgQECAv7+/"
    "n5+fr6+vN6m8vLw8PT09PDzc3d3d3NxcXV3RaLSLi4uzs7OTk5Ojo6ODg4O9vb2dnZ0Nqaytra2s"
    "rCwtLS0sLMzNzc3MzExNTU1MTIxIZWhoaGBgoK+vr6enp6urq62t/YdUWlpav3//1tTU1NDQUFdX"
    "V1NTUyGVsrKykpKSoqKigoKCvLy8rKysDKmkpaWlpKQkJSUlJCTExcXFxMRESSUiIvLz509hYWEh"
    "IaEfP34ICgoKkOrbt2/8/Px8fHy8vLxfv37l4eHhJhUXFxcnJycHBwc7OzsbGxsrK+snUn38+PHD"
    "hw8sLCzv379nZmZ+9+7dmzdvXpPq1atXL1++fPHixfPnz58+ffrkyZPHpHr06NHDhw8fPHhw//79"
    "e/fu3blz5zapbt26xcTEdPPmTUZGRnp6ejo6uhukoqWlpaGhuX79+rVr165evXr58uVLpLp48eKF"
    "CxfOnz9/7ty5M2fOnD59+hSpTp48eeLEiePHjx87duzo0aPU1NRUpKKkpKSgoDh8+DA5OTkZGRkK"
    "hQL/FfIz8i+HDh1C/gt5AfIy5MXIryC/e+TIEeQiyKWQCyKXRS6OvAXyXsg7nj17FnlrZADIMJDB"
    "IENCxnblyhVkkMhQkQEjw0YG/+8ukDtiYGBgJBVym8jN/rvru3fvIpNwn1TItCCT82+WkBl79uzZ"
    "c1Ih04hM5r9Zffv2LTLJzKRCph2ZfGQJ/q3F58+fkaVBFghZJmSxkCX7t3bIIiJLiSwosqzI4iJL"
    "/G+tv3//jiw9YgDEBogZEEv88wZiEsQqiGEQ2yDmQSz0z0tycnKItRCDITZDzIZY7p/3EBMiVkQM"
    "idgSMSdi0X9e1dHRQayLGBixMWJmxNL/vI2YHLE6YnjE9oj5kQggQUDi8C8XSECQmCBhQSKDBAeJ"
    "DxIiJEpIoJBYIeFCIoYEDYnbv9whAURiiIQRiSQSTCSeSEiRqCKBRWKLhBeJMBJkJM7/co0EHIk5"
    "EnYk8kjwkfgjEEBQgAABwQICBwQRCCgQXPzjBgIQBCMITBCkIGBB8IJABkENAhwEOwh8EAQhIEJw"
    "9I9LCKAQTCGwQpCFgAvBFwIxBGUI0BCsIXBDEIeADsHdP+4hAEQwiMAQQSICRgSPCCQRVCLARLCJ"
    "wBNBKAJSBKfTpEIAi2AWgS2CXAS8CH4RCCMoRoCMYBmBM4JoBNQIrrGkQgCOYByBOYJ0BOwI3hHI"
    "I6hHgI9gH4H//5MAJCNIUqgoKagpKY9QUR2jpj5+5MiJo0dPHjt2+vjxMyeOnz1x4tzJk+dPnbp4"
    "+vSlM2cunz175dy5a+fPXT9/nubCBdqLF29cukR/+TLDlSuMV6/evHaV6dq129ev36GhuUtLe+/G"
    "jQd0dA/p6R8x0D9mYHjCyPjs5s3nTEwvbt16efv26zt33ty98/bu3Xf37jHfv8/y4MGHhw8/Pnr0"
    "6fFj1ieP2Z48YX/6lOPZM87nz7lfvPjy8iXPq1dfX7/iff2a/82bb2/fCrx7J8jM/OP9eyEWFuEP"
    "LD8/fBD5+PHXp09irKzinz9LsLFJsrNLc7DLcHDIcnLKcXEpcHMrfvmixMOj/PWrCu9XNV5edT4+"
    "DX5+zW/ftAQE/ggKan//rv3jh84PYV2hn3rCIvo/RQ1EfhmKihv9kjASkzQWkzIRlzaVkDWTlDOX"
    "kreQVrCUUbSUVbaSVbGWU7WRV7NV0LBT1LRX+u2grOWg8sdRRcdJVddZTc9FXR+tYeCqaeT229hN"
    "y8Rdy9Tjj5mntoWXjqW3rpWPnrWvvo2vgZ2fgb2/oUOAkWOgsVOQiUuwKTrEzDXE3C3U3D3MwjPc"
    "0ivCyjvS2ifKxjfa1j/aLiDGLjDWPijOITjeMTTBKSzROTzJJSIJHZmMjk5xjUl1i01zj0v3iM/w"
    "TMz0Ssr0Ts7yTsn2Sc3xTc/1y8jzz8wPyCoIzC4Iyi0MyisKzi8OKSgJLSoNKy4LLymPKC2PLKuI"
    "rKiMqqyKrqqOqa6JramNq6uLr69LqK9PamhIbmxMaWpKa25Ob2nJbG3NamvNbmvLbW/P6+jI7+ws"
    "7Ooq6u4u6ekp7e0p6+2t6Our7O+vGhioGRysHRqqGx5uGBluHBlpHh1tGRtrHR9vn5jomJzsnJrq"
    "np7qmZ7unZnpn50dmJsbmp8fXlgYWVwcW1ocX1qaWF6eWlmZXl2dXVubW1+f39hYxGwsYTDLm5ur"
    "WOza1tY6DofZ3t7c2dna3cHt7m4jDzbI4w3ykPNfABD/0wBAT0Fx69ix+2fPPrl8+cW1a69paZnp"
    "6T8wMrIyMbHfucN1796XBw94Hz7kf/xY8OnTH8+f/3zxQvTVK7HXryXevpV6907m/Xs5FhaFDx+U"
    "Pn1SYWVV+/xZg41Nk4NDi5NTm4tLl5tb78sXAx4eI15eYz4+U35+s2/fLAQELAUFrb9/t/nxw05I"
    "yF5Y2PHnTycRERdRUfSvX65iYu7i4h4SEl6Skt6Skj5SUn7S0v4yMgGysoFycsHy8iHy8qEKCmGK"
    "ihFKSpHKylEqKtEqKrGqqnFqavHq6gkaGokaGkmamim/f6dqaaVpaaX/+ZOhrZ2po5Olo5Otq5uj"
    "p5enr5+vr19gYFBoaFhkaFhsZFRibFxqYlJmYlJualphZlZpZlZlYV5laVFlbVFtY1ltZ1Vjb1Xj"
    "aF3rZFPrYlOLtq1zta1zt6v3sK/3tG/wdmjwcWzwdWz0d2oMcGoKdG4KdmkOcWkORTeHubZEuLZE"
    "urVGubVGu7fGeLTFebTFe7YneLYnenUkeXUke3ek+nSm+XSm+3Zl+HZl+nVl+XVn+3fnBPTkBvTk"
    "BfYUBPYWBvUWBfUVB/eVBPeVhvSXhfaXhw5UhA1Uhg1UhQ9Whw/WRAzWRgzVRQ7VRw43RA03Rg03"
    "RY80R4+0xIy2xoy2xY62x451xCEb/bHO+PGu+PHuhImehInexIm+pMn+pMmBpKnB5Kmh5KnhlOkR"
    "RKnTo6kzY2kz42mzE+mzk+mzUxlz04gy52Yy52ez5uey5uezFxYQ5Swu5i0u5i8uFi4tFS0tFS8t"
    "lS4vly0vly8vV66sVK2sVK+u1qyu1q2u1q+tNaytNa6tNa+vt6yvt66vt21sdGxsdG5sdGEw3RhM"
    "z+Zm3+Zm/+bmABY7iMUOYbHDW1ujW1tjW1vjONwEDjeJw01tb09vb89sb8/t7Mzv7Czs7Czu7i7t"
    "7i7v7q7s7a3u7a3t7a3v72/s72P29zcPDrAHB1sHBzg8fhuP38HjdwmEPQLhAHnYRx75//H/CADH"
    "ATgJwBkAzgNwEYArAFwDgBbJBQCMANwC4C4A9wF4BMATAJ4D8AqANwAwA/ABgE8AsAHAAQA3AF8B"
    "4ANAAIAfAAgDIAqAGACSAMgAIAeAIgAqAKgBoAmAFgA6AOgDYAiACQDmpMN4NqSzeY4AuJDO7HkA"
    "4E06yxdAOtoXCkAE6chfLAAJACQBkEo6GZgFQC4A+QAUAVAKQDkAVQDUAFAPQBMALQC0A9AFQA8A"
    "/QAMAjACwDgAkwDMADAPwCIAKwCsAYABYAuAbQD2AMADQAQAkgF4GEBKgMwRPAbgSQDPAHgOwIsA"
    "XgHwGoC0ANIByAjgLQDvAHgfwEcAPgHwOYAvAXwDIDOALAB+ApANQA4AuQHkAZAPQAEAvwMoDKAo"
    "gGIASgIoDaAcgIoAKgOoBqAmgFoA6gCoB6AhgCYAmgFoCaANgHYAOgLoDKArgB4AegHoC6A/gEEA"
    "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"
    "7Nr2v0O1Eyu48RXc6DJuZAk3tIQbXMT1L+D6FnA987juOVznHK5jZqt1Cts0sVk/hqkZwVQOb5QN"
    "rhf3rxX0ruZ2r2Z1rWR0LKe2LSW1LMU3LcY0LkTWz4fVzgVX//2biG/FjHfZtEfJlGvRlHPBpEPe"
    "hG3OmFXWqEXGsGnqkFHyoEFiv25835/Ynt9R3eoRnSphHYohbfKBrTL+zZK+TeLejaIe9cJudd/R"
    "NQLO1XwOlTx2Fdw2ZRxWpZ8tSj6ZFX0wKWAxyn9vkPtOL/utTtabPxmvf6e91Eh9oZb8XCXxmVLC"
    "E4W4x3Ixj2SiH0pF3peIuCcWdlc05M7P4FtCgUzf/W8K+DHy+9DzetF99bzxxf0Gt+sNThdaDida"
    "Nkfaz/Y0n2xpPlrTsFjSvLe4/s7s+luT66+Nrr0yuPZC/9pz3atPta8+0br6SPPKQ40r99Wu3FO5"
    "fEfp8m2Fy7fkL38EPKwoXrZD/BzkAlwU379QCX2lFuE7+uvbcfHvJySFTkn/PCMnelZB7LySxEUV"
    "qUtqMlc05a9pKV7XVqbVVb2hr05vpMloonXTTPuWhe4dK/17NoYP7Ywe2xs/czR96Wz2xsX8nasl"
    "i5vVJw8bNk9bDm87bl+Hr36O/AFOAoEuP4LRP0PcfoW5i0d4SEV6yUZ7K8T4KMX5qcb7ayQGaiUH"
    "aaeE6KaG6qeHG2ZEGmdFmWbHmOfEWubFW+cn2hYm2RelOBanupSmu5ZluldkeVbmeFfn+tbk+9cW"
    "BtYXBTeUhDaVhjeXR7ZURrdVxbbXxHfUJnbWJ3c1pHY3pvc0Z/S2ZPW15fS35w10Fgx2FQ31lAz3"
    "lo70l48OVI4NVo8P106M1E+ONUyNN01PtsxMtc3OdMzNds3P9yzM9y4s9C0uDCwuDi4tDS0tjSwv"
    "j66sjK2sTKyuTq6uTq+tzayvz5I+kljYwCySPpJY2dxcw2LXsVsbW1ubWzgsbnvr/7gf+WlrG7u5"
    "vYnBbWzg1tdxq2tbK6tby8tbi0tbC4vYuQXs7Dx2ehY7NbM5Ob05PrU5NokZmcAMj2GG/h6p3egf"
    "/j+Habv71zt719q7V1s7V5rblxtbl+ubl2obF6vrFypr58ur50or54rLZwtLZ/KLp3MLp7LzpzJz"
    "J9OzJ1Izx5PTxxJTR+OTR2MTR6LjhyNjhsIiB0LC+gKDe/wDun18O7282t3dW13RLS5OTY72DfY2"
    "dTaWNVZm1ebGlaYG5Ua6pQZ/SnQ1i7TVCn4r52ko5KrKZitLZSqIp8v9SpP5mSwllCghGC/+LfYX"
    "b7QIT+RPrnAhjtDvn4MEPwV8Y/HjY/bhfePF88qD+7kb11M0x2Nn9gcOn+/Zfbpt85HJioXBgpnO"
    "7B2NyZtrRq+u6L+8rPvikvbzi1rPLmg+Pa/++Jzqo7PKD88oPjglf/+k7L0T0nePS945Jn776K9b"
    "R0RuUgszUv1goBSkp/hGd5jvBvlX2kNfaMi4rqM4rgG2qz8eHxZ+cljk6eFfzw6Lvzgs+fKwzCsK"
    "uTcUCm8plN5RqDBTqLNQan6g1PpIqc1KqfuZ0oCNyoidyoSTyoyLyoKbypqH2vYrtT0vtSM/tfM3"
    "aleBI+7fj3kKHff+ecJX9JS/2OlAiTPBUudCZc6Hy16MlL8UrXg5VvlqvOq1RPXryZq0qVo30v/Q"
    "Z+owZOsx5how5RvdKjS5XWx2t9TiXrnVgwqbx5X2z6ocX9a4vKl1Za7z+FDvxdroy9Hkz90c9LUl"
    "lL8tXLA9SqgjRqQzXrw7UaonRbY3TaEvU3kgW20wT3OoUHu4WG+0zHCswmS8ynyixmqy1maq3n66"
    "wWm6CT3T7Dbb6jnX5jPf4b/QGbjYHbLUE77cF7XSH7MyGL86lLQ2kro+mr4xnoWZyN2cKsBOF23N"
    "luLmKrYXqnELNbjFOtxiA26pCbfUgltuwy134Fa6cCs9uNU+3OoAbm0ItzaCWx/DrU/gNqZwGzM4"
    "zBwOs4DbXPp7ZBa7hsNu4LY2//5x8W8atjaw2DXs5gp2c3kTs7i5MY/ZmMWsT2+sT22sTWysjq2v"
    "jqyvDK0tD64t968u9a4uda8udq4sdKwstC3PtyzPNS3NNSzN1i/N1ixMV85Plc1NFM+OF86M5k2P"
    "ZE8NZ0wMpo4PJI/1JYz2xo70RA13RQx1hg60B/W3+fe1+vQ2e/U0uXc3oLvqnTpq7dtrbNuqrVor"
    "zZsrTBrLDOtL9GqLtKsLflfmqZfnqJRmKRVnyBekyeSlSOYkiWUliGTECaXFCKZE8SdFfI0P444N"
    "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"
    "54h9s7O962D5YidxsX16oXVqoXlyvnFivn58rnZstnp0tnJkpnx4pnSIWDw4XUCYzhuYyumfyu6b"
    "zOydTO+ZSO0eT+4aT+wci+8Yi20fjW4biWwdCWsZDmkaCmoYDKgj+NUO+Fb3/6zs86ro9SjrdS/p"
    "+V7U/Q3zSH6nY26Hw592++w228zWT+ktVmnN5ilNpkmNxgkNhvH1+rF1utG1H6NqP4RXa4VWvQ2u"
    "fBNY8cq/XN23TM2n9IV38XPPIpUfhUrfCxS+5cs75z1xypX98kfGPlvaNuvBp0xJqwwJi3Qx0zQR"
    "4xRhw2RB/SR+3cR72gl87+P53sbefRPDqxHNoxbFrRrJ/Sz8jnLYbYVQrqchnLLBnDKBtx4GcEj5"
    "s4v7sYv6sgn/YhX0Ybn3k/muNzO3583bHky3ftxgd7/B6vZ/nv5fkpLVSamvSOmvSZmapOy3pBwt"
    "Ut4HUoE2qUiHVKJHKtMnVRiSqoxJNSakOjNSgwWpyZLUYk1qsyF12JK67Mk9DuReR3LfV3K/C3nA"
    "lTzoRh76QR72JI94k0d9yOO+5Ak/8mQAeSqITAwhz4SRZyPIc1Hk+RjyYhx5KYG8nEReSSGvppFJ"
    "GWRyFvZaI2esktKW11KW1pIWVhPmV+JmV2JmlqOmlyOmlsImFkPGF4NGFwJG5v2G5n0H53wG5rz7"
    "Zz17Z370zLh1EV07p13ap7+2TTm2Tjk0T9o3TtjVj9vWjn2qHrWuHLEqH7YoHTYvHjIrHDTNJxjn"
    "Dhj96TfM6tPP6NNL69VN6dFJ6tZO6PoQ1/k+puNdVMfbiHbNsLbXIa2vAls0/Js1fJs0fBo1vBrU"
    "PerU3WrVXWvUnKvVnKrUHCrV7Cte2pS/tC57aVH60qz4hXHRC8PCF3oFqjr5qh/yVLVyVTVznr/+"
    "81w9+/nLrOfPM54pp6sopCo/SVF+nKQknagoFa8gHqcgEiMvFP2UP+rJ3Ygn3OFyXKGyHCGPWYMe"
    "3wyUYfR/dM1P+rKv9MVfD8/7PDjjLXXSS+qYhyTtDwkaN/EjrmKHXUSpv4occrx/0EGYyl6I0k7w"
    "gI3Afmv+fZb39prz7TG9u9uYd5chz0597h16d7br3N6mzbX1PeeWd7c2a3Jses1OobHOV/+iuPt7"
    "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"
    "yUr/prXeTWsd5k8fmT9ps9h8YLHRYrV9x2r3ls1Ok83+Dbv9K/bPGuyf1Tkc1Di+vLj1RfWW43NO"
    "x2ecTipcTkpcXxVvOyvcdpa/4/Lkjosc9zdZbtfHPK6PeL5L83x/yOv2gNdN6q67xN0f4nw/xPg8"
    "RO953L/nKczvKSToJSjsLSDiLSD+U1Dyp+ADHyFpH6HHv4TlfO8/9b2v8FtE6bfIMz9RVT+xl/5i"
    "6gHirwPENQMk3gVKvA+U1A6S0g2S0g9+YBD8wCjkgUnIA9OQh+ahDy1DH1qFSX8Kk7YNl7YLf/Q5"
    "4tGXiEeOETJfI2VcImW+RT3+HvXYPfrxj2hZz2hZrxjZnzGyv2LlfGPl/OLkAuKeBMY/CY5/Ehr/"
    "NCzhaUTC06hE+ehE+dgk+fgkhYQkhaRkheRkxZSk9VdqolJqglJavFJanFJarHJ6zPorI1olI0ol"
    "M1IlM+JZZvizrDDs9Tw79Hl2yPPsYNU/Qap/AlVzAl7kBKjlBmjkBbzJX78KeF8UpF0crFsSrF8a"
    "YlQWalIeZlYRblkVYV0daVsTZV8b9aUu2qk+xqUh1rUxzr0p3qMlwas1yact2bc9xb8jLbAzPaQr"
    "M6w7K7L3T0xfbnx/ftJAYSqhOGOwNHuoPHe4smCkunistmy8vnKisWayuX6qtWm6o5XY1THT0z3X"
    "3zc/SFgYGV4cH1uamlyeIa7Mz60uLZJWVm7e9LzB4s3I/ouB0+/a7YCrPMFX+MIu8UfQC0VfFIm7"
    "IJZwXjL53MO0M48yTstmn3qae0Ih/7hy0bFnpbQvymnUq468rqXWrD+k1XRQu5VSp/2Aftc+o969"
    "Jv17zAd3W43s/DS2w25y+2fiVsfZLc4Lm12XKdxWwYMMXivwcw5+TcLvYfDvg8AOCG6C0BoIL4PI"
    "AojOhthUiI+HxEhIDobU35DuBZnfIdsJcuwgzxIKjKFIF0q0oOwVRcXLrZVqOyrV9lapU1ZrUNe8"
    "oq19faL29Zm6NxfqNekb3l5rfHej8R1LkxZH8/vbLR94W7X5W7WF2j6KtetIdeg+6tST69RT6NJX"
    "6TZ40WOo3muo2Wv0vs9Yp9/EYMDUhGBmQTD7NGhuN2ThOGzpMmLlNmLlOWrtM/bJb9wmaMI2dMI2"
    "atIubso+afpzGtEhi+iQO/OlcNaxZM6pcv5r7fzXxgXn1kWXzqVvvcuuhGXX4ZXvE6tuxDX3edKP"
    "ZdIPEtmDTPJYW/2xsuS+tOC2MPt9btp1ZtJlesx5cuTr+KDT6IDjcO+XwW6HgY7Pfa32Pc12XQ22"
    "HXU2bdWfWiqtmsos64st6grNa/LMqnJMK7JMyjKMS1KNipIN8xMM8mL1c6J1syN0MsM+pgdrpwZ+"
    "SPZ7n+CrFf/zXazn2+gfmpHf34R/ex369VWwo3rAZzV/25e/P734Zan60/y5l8kzDyMVd31lV12l"
    "b9qKzloKTm+ffnn95LOGnN1LWRvVx1YqMpZKj8zlpS0UZKyUZT89e2r7QtFOTdnh1XNHzZdf36m7"
    "fHjt+vGtm57WDwNtT2NdbzMDHwsjX2tTPxuLAHurIAebECf7UGeHCFenKHeXGA/XOG/3BB/PpN/e"
    "Kf6/0oL8MkICssKD/0SF5cZE5MdHFyTGFacklKQll2emVf7JqM7Nri3IrS8qaCgtai4vba2qaK+p"
    "6qyv7W5q6G1p6m9vHejsGOrpGu7rHSMMTAwNTo2OECfGZ6cm52aIi3Nzy4sLq8vLpLW1jfpPRbHp"
    "/KYtrJu3CW/Z8XTrLs1te8y27/u240DgTqrkXYdKdx9u30Mzsfcoed/x5f0n5w6cnqI8O0p1nnCQ"
    "rucQfTv15abDV2uPXK+gYSymZco7ypx1jDX1OHviiVsxJ7nCT90JOs3z+zSv99l77ucEXM4LfTkv"
    "bEsnanlR3IReUp9eSvuy9NsrMhpXZVWvyildl3/CoPiIUVmSUUWESVXg5kteZnUuZg021jdMbG+v"
    "sWvRs78/d+vjSU5dWi79Q1wG++8Y7+I25eEx5+W15OW14uOzuXfPjp//s4DAF0FBJyEhZ2Hhb8LC"
    "riIi7qKiHmJiXuLiPyUkfklK/paS8peSCnj4MFhaOvTRo3AZmcjHj6NlZWPl5OLl5BKePk2Wl09V"
    "UEhXVMxUUspWVs5RUclTUcl//rxIVbXkxYuyly8r1NSq1NVrNDTqNDTqX79uevOmRVOz7e3bjnfv"
    "urS0et6/73v/vl9be/Djx2EdnVFd3XE9vUl9/WkDgxkDg1kjowVj4yUTkxVT0zUzM7K5+ZqFxZKl"
    "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"
    "+fnc+vrg+vvl+/vq/Pzu/P3z/f73/kcqkvk="
)
def _load_luts():
    raw = zlib.decompress(base64.b64decode(_LUT_B64))
    return np.frombuffer(raw, dtype=np.uint8).reshape(_N_CMAPS, 256, 3).copy()
_LUTS = _load_luts()
_LUT_IDX = {name: i for i, name in enumerate(_ALL_CMAPS)}
def get_lut(name):
    return _LUTS[_LUT_IDX.get(name, _LUT_IDX["inferno"])]
def apply_lut(magnitude_01, cmap_name):
    lut = get_lut(cmap_name)
    return lut[(np.clip(magnitude_01, 0, 1) * 255).astype(np.uint8)]


# ---------------------------------------------------------------------------
# Gaussian filter  (replaces scipy.ndimage.gaussian_filter)
# ---------------------------------------------------------------------------

def gaussian_filter_np(arr, sigma):
    if sigma <= 0:
        return arr
    radius  = max(1, int(math.ceil(3 * sigma)))
    x       = np.arange(-radius, radius + 1, dtype=float)
    kernel  = np.exp(-x**2 / (2 * sigma**2))
    kernel /= kernel.sum()
    arr = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), axis=1, arr=arr)
    arr = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), axis=0, arr=arr)
    return arr


# ---------------------------------------------------------------------------
# Point-in-polygon  (replaces matplotlib.path.Path.contains_points)
# ---------------------------------------------------------------------------

def _points_in_polygon(coords, polygon):
    result = np.zeros(len(coords), dtype=bool)
    cx, cy = coords[:, 0], coords[:, 1]
    n = len(polygon); j = n - 1
    for i in range(n):
        xi, yi = polygon[i]; xj, yj = polygon[j]
        cond1 = (cy > yi) != (cy > yj)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_int = (xj - xi) * (cy - yi) / (yj - yi) + xi
        result ^= cond1 & (cx < x_int)
        j = i
    return result


# ---------------------------------------------------------------------------
# Input-file parser
# ---------------------------------------------------------------------------

DEFAULTS = {
    "n":                    42,
    "mc_cycles":            1,
    "crop_shape":           "faded",
    "crop_radius_factor":   1.0,
    "output_folder":        ".",
    "fft_zoom_factor":      0.3,
    "gaussian_sigma":       1.0,
    "colormap":             "afmhot",
    "intensity_low":        0.0,
    "intensity_high":       100.0,
    "export_size":          3000,
    "save_mc_energy_plot":  True,
    "save_diff_ff":         True,
    "save_avg_structure":   False,
}

VALID_CROP_SHAPES = {"none", "circle", "faded", "ellipse", "square", "pentagonal"}


def parse_input_file(path: str) -> dict:
    """
    Parse a key = value input file.

    Special multi-value keys:
        tile        = <occupancy>, <image_path>
        interaction = <from_index>, <to_index>, <dx>, <dy>, <energy>

    Tile indices in interactions are 1-based (matching tile order in the file).
    Lines starting with # are comments; blank lines are ignored.
    """
    params = dict(DEFAULTS)
    params["tiles"]        = []   # list of (occupancy, path)
    params["interactions"] = []   # list of (i, j, dx, dy, energy)  — 0-based indices

    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                _warn(f"Line {lineno}: no '=' found — skipped: {raw.rstrip()}")
                continue

            key, _, rest = line.partition("=")
            key  = key.strip().lower()
            rest = rest.strip()

            if key == "tile":
                parts = [p.strip() for p in rest.split(",", 1)]
                if len(parts) != 2:
                    _warn(f"Line {lineno}: 'tile' needs 'occupancy, path' — skipped.")
                    continue
                try:
                    occ = float(parts[0])
                except ValueError:
                    _warn(f"Line {lineno}: occupancy '{parts[0]}' is not a number — skipped.")
                    continue
                params["tiles"].append((occ, parts[1]))

            elif key == "interaction":
                parts = [p.strip() for p in rest.split(",")]
                if len(parts) != 5:
                    _warn(f"Line {lineno}: 'interaction' needs 5 comma-separated values — skipped.")
                    continue
                try:
                    i_from = int(parts[0]) - 1   # convert to 0-based
                    j_to   = int(parts[1]) - 1
                    dx     = int(float(parts[2]))
                    dy     = int(float(parts[3]))
                    energy = float(parts[4])
                except ValueError as exc:
                    _warn(f"Line {lineno}: bad interaction values ({exc}) — skipped.")
                    continue
                params["interactions"].append((i_from, j_to, dx, dy, energy))

            elif key == "n":
                params["n"] = _int(key, rest, lineno, DEFAULTS["n"])

            elif key == "mc_cycles":
                params["mc_cycles"] = _int(key, rest, lineno, DEFAULTS["mc_cycles"])

            elif key == "export_size":
                params["export_size"] = _int(key, rest, lineno, DEFAULTS["export_size"])

            elif key in ("crop_radius_factor", "fft_zoom_factor",
                         "gaussian_sigma", "intensity_low", "intensity_high"):
                params[key] = _float(key, rest, lineno, DEFAULTS[key])

            elif key == "crop_shape":
                val = rest.lower()
                if val not in VALID_CROP_SHAPES:
                    _warn(f"Line {lineno}: unknown crop_shape '{rest}'. "
                          f"Valid options: {sorted(VALID_CROP_SHAPES)}. Using default.")
                else:
                    params["crop_shape"] = val

            elif key == "colormap":
                if rest not in _ALL_CMAPS:
                    _warn(f"Line {lineno}: colormap '{rest}' not recognised. "
                          "Using default 'afmhot'.")
                else:
                    params["colormap"] = rest

            elif key == "output_folder":
                params["output_folder"] = rest

            elif key in ("save_mc_energy_plot", "save_diff_ff", "save_avg_structure"):
                params[key] = _bool(key, rest, lineno, DEFAULTS[key])

            else:
                _warn(f"Line {lineno}: unknown key '{key}' — skipped.")

    return params


# ---------------------------------------------------------------------------
# Small parsing helpers
# ---------------------------------------------------------------------------

def _warn(msg):
    print(f"  [WARNING] {msg}", file=sys.stderr)


def _int(key, val, lineno, default):
    try:
        return int(val)
    except ValueError:
        _warn(f"Line {lineno}: '{key}' value '{val}' is not an integer. Using {default}.")
        return default


def _float(key, val, lineno, default):
    try:
        return float(val)
    except ValueError:
        _warn(f"Line {lineno}: '{key}' value '{val}' is not a number. Using {default}.")
        return default


def _bool(key, val, lineno, default):
    if val.lower() in ("true", "yes", "1"):
        return True
    if val.lower() in ("false", "no", "0"):
        return False
    _warn(f"Line {lineno}: '{key}' value '{val}' is not True/False. Using {default}.")
    return default


# ---------------------------------------------------------------------------
# Image-processing helpers  (unchanged logic from the GUI version)
# ---------------------------------------------------------------------------

def crop_to_square_center(img: Image.Image) -> Image.Image:
    """Crop a rectangular image to a centred square."""
    w, h = img.size
    if w == h:
        return img
    side   = min(w, h)
    left   = (w - side) // 2
    top    = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def pad_to_minimum(arr: np.ndarray, target: int = 5000) -> np.ndarray:
    """Centre-pad a (square) array to at least target×target pixels."""
    h, w = arr.shape[:2]
    if h >= target and w >= target:
        return arr
    new_h, new_w = max(h, target), max(w, target)
    pt = (new_h - h) // 2;  pb = new_h - h - pt
    pl = (new_w - w) // 2;  pr = new_w - w - pl
    kw = dict(mode="constant", constant_values=0)
    if arr.ndim == 2:
        return np.pad(arr, ((pt, pb), (pl, pr)), **kw)
    return np.pad(arr, ((pt, pb), (pl, pr), (0, 0)), **kw)


def apply_mask_crop(image: Image.Image, crop_radius_factor: float, shape: str) -> np.ndarray:
    """Return a numpy array with the requested mask/crop applied."""
    arr         = np.array(image.convert("L"))
    h, w        = arr.shape
    cx, cy      = w // 2, h // 2
    max_r       = min(w, h) // 2
    radius      = int(max_r * crop_radius_factor)
    Y, X        = np.ogrid[:h, :w]

    if shape == "none":
        return arr

    if shape == "circle":
        mask = (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2
        return np.where(mask, arr, 0)

    if shape == "faded":
        mask        = (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2
        radial_dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        r_start     = radius * 0.8
        sigma       = (radius - r_start) / 2
        fade        = np.ones_like(arr, dtype=float)
        zone        = (radial_dist > r_start) & (radial_dist <= radius)
        fade[zone]  = np.exp(-((radial_dist[zone] - r_start) ** 2) / (2 * sigma ** 2))
        return (arr * fade * mask).astype(np.uint8)

    if shape == "ellipse":
        rx   = min(radius, w // 2 - 1)
        ry   = min(int(radius * 0.75), h // 2 - 1)
        mask = ((X - cx) ** 2 / rx ** 2) + ((Y - cy) ** 2 / ry ** 2) <= 1

    elif shape == "square":
        mask = (np.abs(X - cx) + np.abs(Y - cy)) <= radius

    elif shape == "pentagonal":
        angles  = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        polygon = [(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles]
        xx, yy  = np.meshgrid(np.arange(w), np.arange(h))
        coords  = np.column_stack((xx.ravel(), yy.ravel())).astype(float)
        mask    = _points_in_polygon(coords, polygon).reshape((h, w))

    else:
        mask = np.ones_like(arr, dtype=bool)

    return np.where(mask, arr, 0)


def apply_fft_processing(magnitude: np.ndarray, params: dict) -> np.ndarray:
    """Apply smoothing, zoom, percentile clip, and intensity-range rescaling."""
    sigma = float(params["gaussian_sigma"])
    if sigma > 0:
        magnitude = gaussian_filter_np(magnitude, sigma)

    zoom   = max(0.01, min(float(params["fft_zoom_factor"]), 1.0))
    h, w   = magnitude.shape
    zh, zw = int(h * zoom), int(w * zoom)
    sy, sx = (h - zh) // 2, (w - zw) // 2
    magnitude = magnitude[sy:sy + zh, sx:sx + zw]

    magnitude = np.clip(magnitude, 0, np.percentile(magnitude, 99))
    if magnitude.max() != 0:
        magnitude /= magnitude.max()

    lo = float(params["intensity_low"])  / 100.0
    hi = float(params["intensity_high"]) / 100.0
    magnitude = np.clip(magnitude, lo, hi)
    magnitude = (magnitude - lo) / (hi - lo + 1e-8)
    return magnitude


# ---------------------------------------------------------------------------
# File-naming helper
# ---------------------------------------------------------------------------

def next_available_index(folder: str, prefix: str) -> int:
    existing = []
    for fname in os.listdir(folder):
        if fname.startswith(prefix + "_"):
            try:
                existing.append(int(fname.split("_")[1].split(".")[0]))
            except Exception:
                pass
    if not existing:
        return 1
    for expected, num in enumerate(sorted(existing), start=1):
        if num != expected:
            return expected
    return len(existing) + 1


# ---------------------------------------------------------------------------
# Energy calculation
# ---------------------------------------------------------------------------

def calculate_energy(g: np.ndarray, name_to_idx: dict, interactions: list) -> float:
    energy   = 0.0
    idx_grid = np.vectorize(lambda nm: name_to_idx[nm])(g)
    rows, cols = g.shape
    for (i_from, j_to, dx, dy, e_val) in interactions:
        dx_np = -dy
        dy_np =  dx
        i0 = max(0, -dx_np);  i1 = min(rows - 1, rows - 1 - dx_np)
        j0 = max(0, -dy_np);  j1 = min(cols - 1, cols - 1 - dy_np)
        if i0 > i1 or j0 > j1:
            continue
        src = idx_grid[i0:i1 + 1, j0:j1 + 1]
        nb  = idx_grid[i0 + dx_np:i1 + dx_np + 1, j0 + dy_np:j1 + dy_np + 1]
        energy += e_val * np.count_nonzero((src == i_from) & (nb == j_to))
    return energy


# ---------------------------------------------------------------------------
# Optional-output helpers
# ---------------------------------------------------------------------------

def compute_diff_ff(paths: list, min_size: int, params: dict) -> np.ndarray:
    """
    Compute |FT(tile1) − FT(tile2)|² using the first two valid tile paths,
    then apply the same FFT processing (smoothing, zoom, intensity range)
    as the main pattern.  Returns a processed magnitude array (0–1).
    """
    TARGET = 5000
    imgs = []
    for p in paths[:2]:
        img = crop_to_square_center(Image.open(p).convert("L"))
        if img.size[0] != min_size:
            img = img.resize((min_size, min_size), Image.LANCZOS)
        arr   = np.array(img, dtype=float)
        pad_y = (TARGET - arr.shape[0]) // 2
        pad_x = (TARGET - arr.shape[1]) // 2
        padded = np.pad(
            arr,
            ((pad_y, TARGET - arr.shape[0] - pad_y),
             (pad_x, TARGET - arr.shape[1] - pad_x)),
            mode="constant", constant_values=0,
        )
        imgs.append(padded)

    F1   = fft.fftshift(fft.fft2(imgs[0]))
    F2   = fft.fftshift(fft.fft2(imgs[1]))
    diff = np.abs(F2 - F1) ** 2
    return apply_fft_processing(diff, params)


def compute_avg_structure(paths: list, occs: list, min_size: int) -> Image.Image:
    """
    Return a greyscale PIL Image of the occupancy-weighted average tile.
    """
    arrays = []
    for p in paths:
        img = crop_to_square_center(Image.open(p).convert("L"))
        if img.size[0] != min_size:
            img = img.resize((min_size, min_size), Image.LANCZOS)
        arrays.append(np.array(img, dtype=float))

    avg  = sum(arr * occ for arr, occ in zip(arrays, occs))
    pmin, pmax = avg.min(), avg.max()
    if pmax > pmin:
        avg = (avg - pmin) / (pmax - pmin)
    return Image.fromarray((avg * 255).astype(np.uint8), mode="L")




def run(params: dict):
    # ── Validate / clamp ────────────────────────────────────────────────────
    n      = max(1, min(params["n"], 1000))
    cycles = max(0, params["mc_cycles"])
    tiles_raw  = params["tiles"]
    interactions_raw = params["interactions"]

    if not tiles_raw:
        sys.exit("ERROR: No tiles defined. Add at least one 'tile = occupancy, path' line.")

    # Normalise occupancies
    occs  = [t[0] for t in tiles_raw]
    paths = [t[1] for t in tiles_raw]
    total = sum(occs)
    if total <= 0:
        sys.exit("ERROR: All tile occupancies are zero.")
    occs = [o / total for o in occs]

    # Check files exist
    for p in paths:
        if not os.path.isfile(p):
            sys.exit(f"ERROR: Tile image not found: {p}")

    # Check for duplicate tiles
    print("Loading tiles...")
    loaded_arrays = []
    for p in paths:
        img = crop_to_square_center(Image.open(p).convert("RGB"))
        loaded_arrays.append((p, np.array(img)))

    for i in range(len(loaded_arrays)):
        for j in range(i + 1, len(loaded_arrays)):
            if np.array_equal(loaded_arrays[i][1], loaded_arrays[j][1]):
                sys.exit(
                    f"ERROR: Tiles appear identical:\n"
                    f"  {loaded_arrays[i][0]}\n  {loaded_arrays[j][0]}\n"
                    "Please remove one of them."
                )

    # Find smallest square tile size; resize all to that
    sizes     = [la[1].shape[0] for la in loaded_arrays]
    min_size  = min(sizes)
    tile_names = []
    tiles      = {}
    used_occs  = []

    for occ, (p, _) in zip(occs, loaded_arrays):
        img  = crop_to_square_center(Image.open(p).convert("RGB"))
        if img.size[0] != min_size:
            img = img.resize((min_size, min_size), Image.LANCZOS)
        name = os.path.splitext(os.path.basename(p))[0]
        tile_names.append(name)
        tiles[name] = img
        used_occs.append(occ)

    # Validate interaction tile indices
    n_tiles = len(tile_names)
    interactions = []
    for (i_from, j_to, dx, dy, energy) in interactions_raw:
        if not (0 <= i_from < n_tiles and 0 <= j_to < n_tiles):
            _warn(f"Interaction ({i_from+1}, {j_to+1}, {dx}, {dy}, {energy}): "
                  f"tile index out of range (1..{n_tiles}) — skipped.")
            continue
        interactions.append((i_from, j_to, dx, dy, energy))

    # ── Build initial grid ──────────────────────────────────────────────────
    print(f"Building {n}×{n} grid...")
    tile_counts = [int(n * n * occ) for occ in used_occs]
    while sum(tile_counts) < n * n:
        tile_counts[-1] += 1

    grid_flat = []
    for name, count in zip(tile_names, tile_counts):
        grid_flat.extend([name] * count)
    random.shuffle(grid_flat)
    grid = np.array(grid_flat).reshape((n, n))

    # ── MC simulation ───────────────────────────────────────────────────────
    name_to_idx  = {name: idx for idx, name in enumerate(tile_names)}
    mc_energies  = []
    out_dir      = params["output_folder"]
    os.makedirs(out_dir, exist_ok=True)

    print(f"Running MC simulation ({cycles} cycles)...")
    if cycles >= 2:
        log_path = os.path.join(out_dir, "MC_global_energy_log.txt")
        with open(log_path, "w") as log_file:
            log_file.write("cycle\told_energy\tnew_energy\tcurrent_energy\n")

            for cycle in range(cycles):
                i1, j1 = random.randint(0, n - 1), random.randint(0, n - 1)
                i2, j2 = random.randint(0, n - 1), random.randint(0, n - 1)

                old_energy = calculate_energy(grid, name_to_idx, interactions)
                grid[i1, j1], grid[i2, j2] = grid[i2, j2], grid[i1, j1]
                new_energy = calculate_energy(grid, name_to_idx, interactions)

                if new_energy < old_energy:
                    current_energy = new_energy
                else:
                    grid[i1, j1], grid[i2, j2] = grid[i2, j2], grid[i1, j1]
                    current_energy = old_energy

                log_file.write(
                    f"{cycle + 1}\t\t{old_energy}\t\t{new_energy}\t\t{current_energy}\n"
                )
                mc_energies.append(current_energy)

                if cycles >= 100 and cycle % max(1, cycles // 100) == 0:
                    pct = int((cycle / cycles) * 100)
                    print(f"  {pct}%", end="\r", flush=True)

        if cycles >= 100:
            print("  100%")
    else:
        print("  MC simulation skipped (cycles < 2).")

    # ── Compose tiling image ────────────────────────────────────────────────
    print("Composing tiling image...")
    total_pixels = (n * min_size) ** 2
    MAX_PIXELS   = 100_000_000

    final_img = Image.new("RGB", (n * min_size, n * min_size))
    for i in range(n):
        for j in range(n):
            final_img.paste(tiles[grid[i, j]], (j * min_size, i * min_size))

    if total_pixels > MAX_PIXELS:
        print(f"  Image has {total_pixels:,} pixels — resizing to 100 Mpx automatically.")
        final_img = final_img.resize((10_000, 10_000), Image.LANCZOS)

    TARGET_PIXELS = 5000
    final_img_resized = final_img.resize((TARGET_PIXELS, TARGET_PIXELS), Image.LANCZOS)

    # ── FFT ─────────────────────────────────────────────────────────────────
    print("Computing FFT...")
    crop_shape  = params["crop_shape"]
    crop_radius = float(params["crop_radius_factor"])
    cropped     = apply_mask_crop(final_img, crop_radius, crop_shape)
    cropped_arr = pad_to_minimum(
        np.array(Image.fromarray(cropped), dtype=np.uint8), target=5000
    )

    fft_shifted      = fft.fftshift(fft.fft2(cropped_arr))
    fft_magnitude_raw = np.abs(fft_shifted) ** 2

    magnitude = apply_fft_processing(fft_magnitude_raw.copy(), params)

    # ── Save outputs ─────────────────────────────────────────────────────────
    export_size = int(params["export_size"])
    TARGET_DPI  = 600
    timestamp   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    signature   = (
        f"Fourier Tiler | Saved on {timestamp} | "
        "Designed and developed by Stefano Canossa, supported by EU4MOFs COST action."
    )

    meta = PngImagePlugin.PngInfo()
    meta.add_text("FourierTilerSignature", signature)

    # Tiling
    idx_img   = next_available_index(out_dir, "Image")
    img_path  = os.path.join(out_dir, f"Image_{idx_img:02d}.png")
    tiling_export = final_img_resized.resize((export_size, export_size), Image.LANCZOS)
    tiling_export.save(img_path, dpi=(TARGET_DPI, TARGET_DPI), pnginfo=meta)
    print(f"  Saved tiling        →  {img_path}")

    # FFT intensity
    cmap_name = params["colormap"]
    rgb       = apply_lut(magnitude, cmap_name)
    fft_img   = Image.fromarray(rgb, mode="RGB").resize(
        (export_size, export_size), Image.LANCZOS
    )
    idx_int  = next_available_index(out_dir, "Intensity")
    fft_path = os.path.join(out_dir, f"Intensity_{idx_int:02d}.png")
    fft_img.save(fft_path, dpi=(TARGET_DPI, TARGET_DPI), pnginfo=meta)
    print(f"  Saved FFT           →  {fft_path}")

    # ── Difference form factor |f1 − f2|² ────────────────────────────────────
    if params["save_diff_ff"]:
        if len(paths) < 2:
            _warn("save_diff_ff = True but fewer than 2 tiles are loaded — skipped.")
        else:
            print("Computing difference form factor...")
            diff_mag  = compute_diff_ff(paths, min_size, params)
            diff_rgb  = apply_lut(diff_mag, cmap_name)
            diff_img  = Image.fromarray(diff_rgb, mode="RGB").resize(
                (export_size, export_size), Image.LANCZOS
            )
            idx_diff  = next_available_index(out_dir, "DiffFF")
            diff_path = os.path.join(out_dir, f"DiffFF_{idx_diff:02d}.png")
            diff_img.save(diff_path, dpi=(TARGET_DPI, TARGET_DPI), pnginfo=meta)
            print(f"  Saved |f1−f2|²      →  {diff_path}")

    # ── Average structure ─────────────────────────────────────────────────────
    if params["save_avg_structure"]:
        print("Computing average structure...")
        avg_img  = compute_avg_structure(paths, used_occs, min_size)
        avg_img  = avg_img.resize((export_size, export_size), Image.LANCZOS)
        idx_avg  = next_available_index(out_dir, "AvgStructure")
        avg_path = os.path.join(out_dir, f"AvgStructure_{idx_avg:02d}.png")
        avg_img.save(avg_path, dpi=(TARGET_DPI, TARGET_DPI), pnginfo=meta)
        print(f"  Saved avg structure →  {avg_path}")

    # ── MC energy plot ────────────────────────────────────────────────────────
    has_nonzero = any(e != 0 for (_, _, _, _, e) in interactions)
    if params["save_mc_energy_plot"]:
        if not (has_nonzero and len(mc_energies) >= 2):
            _warn("save_mc_energy_plot = True but no nonzero interactions or fewer than "
                  "2 MC cycles ran — energy plot skipped.")
        else:
            W, H = 600, 400
            pad_l, pad_r, pad_t, pad_b = 72, 20, 44, 70
            pw = W - pad_l - pad_r
            ph = H - pad_t - pad_b
            mn, mx = min(mc_energies), max(mc_energies)
            if mx == mn: mx = mn + 1.0
            plot_img  = Image.new("RGB", (W, H), "white")
            draw      = ImageDraw.Draw(plot_img)
            # Grid + Y ticks
            for i in range(5):
                y   = pad_t + ph - int(i * ph / 4)
                val = mn + (mx - mn) * i / 4
                draw.line([(pad_l, y), (pad_l + pw, y)], fill="#cccccc")
                draw.text((pad_l - 4, y), f"{val:.4g}", fill="black", anchor="rm")
            # Axes box
            draw.rectangle([pad_l, pad_t, pad_l + pw, pad_t + ph], outline="black")
            # Labels
            draw.text((W // 2, pad_t // 2), "Global energy plot", fill="black", anchor="mm")
            draw.text((W // 2, H - 8), "MC cycle number", fill="black", anchor="mm")
            draw.text((10, pad_t + ph // 2), "Global energy", fill="black", anchor="mm")
            # Data line
            if len(mc_energies) >= 2:
                pts = []
                for i, e in enumerate(mc_energies):
                    x = pad_l + int(i / (len(mc_energies) - 1) * pw)
                    y = pad_t + ph - int((e - mn) / (mx - mn) * ph)
                    pts.append((x, y))
                for i in range(len(pts) - 1):
                    draw.line([pts[i], pts[i + 1]], fill="#2F24D8", width=2)
            idx_en    = next_available_index(out_dir, "MC_energy")
            plot_path = os.path.join(out_dir, f"MC_energy_{idx_en:02d}.png")
            plot_img.save(plot_path, dpi=(150, 150))
            print(f"  Saved MC energy     →  {plot_path}")

    print(f"\nAll done. Output folder: {os.path.abspath(out_dir)}")


# ---------------------------------------------------------------------------
# Template writer
# ---------------------------------------------------------------------------

TEMPLATE = """\
# ============================================================
#  Fourier Tiler — Input file
#  Usage: python FourierTiler.py Input.txt
#
#  Lines beginning with # are comments.
#  Key names are case-insensitive.
# ============================================================


# ── Grid ─────────────────────────────────────────────────────
# Side length of the square tiling in number of tiles.
# A value of 42 produces a 42×42 mosaic (1764 tiles total).
n = 42

# Number of Monte Carlo swap cycles.
# Use 0 for a purely random arrangement.
mc_cycles = 1000

# Shape used to crop the tiling before the FFT is computed.
# Options: none | circle | faded | ellipse | square | pentagonal
# "faded" (default) applies a Gaussian-tapered circular window
# that reduces edge artefacts in the diffraction pattern.
crop_shape = faded

# Fraction of the half-width used as the crop radius (0 < value ≤ 1).
crop_radius_factor = 1.0


# ── Output ───────────────────────────────────────────────────
# Folder where all output files are written (created if absent).
output_folder = ./output


# ── FFT / visualisation ──────────────────────────────────────
# Fraction of the FFT to display / export (centred zoom).
# 1.0 = full pattern; 0.3 = central 30 %.
fft_zoom_factor = 0.3

# Standard deviation of the Gaussian smoothing applied to the
# intensity pattern. Use 0 for no smoothing.
gaussian_sigma = 1.0

# Matplotlib colormap name for the intensity pattern.
# Examples: afmhot, inferno, viridis, magma, hot, gray, jet ...
colormap = afmhot

# Intensity-range clipping (percentages, 0 – 100).
# Equivalent to the two slider handles in the GUI.
intensity_low  = 0
intensity_high = 100

# Side length (pixels) of the exported PNG files.
# Allowed values: 1000 | 2000 | 3000 | 4000 | 5000
export_size = 3000

# ── Optional outputs ──────────────────────────────────────────
# Save the MC global-energy plot (PNG).
# Requires nonzero interaction energies and at least 2 MC cycles.
save_mc_energy_plot = True

# Save the squared difference form factor |FT(tile1) − FT(tile2)|²  (PNG).
# Uses the first two tiles listed below. Requires at least 2 tiles.
save_diff_ff = True

# Save the occupancy-weighted average structure (PNG).
save_avg_structure = False


# ── Tiles ────────────────────────────────────────────────────
# Format:  tile = <occupancy>, <path/to/image>
# Occupancies are normalised automatically if they do not sum to 1.
# Any non-square image is centre-cropped to a square.
# Images of different sizes are resized to the smallest one.
# Add or remove lines freely.
tile = 0.5, path/to/tile1.png
tile = 0.5, path/to/tile2.png


# ── Pair interactions ────────────────────────────────────────
# Format:  interaction = <from_tile>, <to_tile>, <dx>, <dy>, <energy>
#
#  from_tile / to_tile : 1-based tile index (order as listed above).
#  dx, dy              : displacement vector components (integers).
#  energy              : real number; negative = favoured, positive = disfavoured.
#
# No symmetry is applied automatically — add each direction explicitly.
# Comment out or delete these lines if no interactions are needed.
#
# Example — tile 1 dislikes tile 2 along x and y, likes it on diagonals:
# interaction = 1, 2,  1,  0,  10.0
# interaction = 1, 2,  0,  1,  10.0
# interaction = 1, 2, -1,  0,  10.0
# interaction = 1, 2,  0, -1,  10.0
# interaction = 1, 2,  1,  1,  -2.0
# interaction = 1, 2,  1, -1,  -2.0
# interaction = 1, 2, -1,  1,  -2.0
# interaction = 1, 2, -1, -1,  -2.0
"""


def write_template(dest: str = "Input.txt"):
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write(TEMPLATE)
    print(f"Template input file written to: {os.path.abspath(dest)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Handle --template flag
    if "--template" in sys.argv:
        dest = "Input.txt"
        for arg in sys.argv[1:]:
            if not arg.startswith("-"):
                dest = arg
                break
        write_template(dest)
        return

    # Resolve input file
    if len(sys.argv) >= 2 and not sys.argv[1].startswith("-"):
        input_path = sys.argv[1]
    else:
        input_path = "Input.txt"

    if not os.path.isfile(input_path):
        print(
            f"ERROR: Input file not found: '{input_path}'\n\n"
            "Usage:\n"
            "  python FourierTiler.py Input.txt\n"
            "  python FourierTiler.py --template   (writes a template Input.txt)",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Reading parameters from: {input_path}")
    params = parse_input_file(input_path)

    # Echo key settings
    print(
        f"\n  Grid              : {params['n']}×{params['n']}\n"
        f"  MC cycles         : {params['mc_cycles']}\n"
        f"  Crop shape        : {params['crop_shape']}\n"
        f"  Colormap          : {params['colormap']}\n"
        f"  Export size       : {params['export_size']}×{params['export_size']} px\n"
        f"  Output folder     : {params['output_folder']}\n"
        f"  Tiles             : {len(params['tiles'])}\n"
        f"  Interactions      : {len(params['interactions'])}\n"
        f"  Save MC energy    : {params['save_mc_energy_plot']}\n"
        f"  Save |f1−f2|²     : {params['save_diff_ff']}\n"
        f"  Save avg structure: {params['save_avg_structure']}\n"
    )

    run(params)


if __name__ == "__main__":
    main()
