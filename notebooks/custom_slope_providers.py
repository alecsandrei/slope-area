from __future__ import annotations

from slope_area._typing import SlopeProvider, StrPath
from slope_area.geomorphometry import compute_3x3_slope


class MaxSlope(SlopeProvider):  # maximum slope (Travis et al. 1975)
    def get_slope(self, dem: StrPath, out_file: StrPath) -> StrPath:
        return compute_3x3_slope(dem, out_file, method=0)


class MaxTSlope(SlopeProvider):  # maximum triangle slope (Tarboton 1997)
    def get_slope(self, dem: StrPath, out_file: StrPath) -> StrPath:
        return compute_3x3_slope(dem, out_file, method=1)


class LsqPlane(
    SlopeProvider
):  # least squares fitted plane (Horn 1981, Costa-Cabral & Burgess 1996)
    def get_slope(self, dem: StrPath, out_file: StrPath) -> StrPath:
        return compute_3x3_slope(dem, out_file, method=2)


class PolynomE(SlopeProvider):  # 6 parameter 2nd order polynom (Evans 1979)
    def get_slope(self, dem: StrPath, out_file: StrPath) -> StrPath:
        return compute_3x3_slope(dem, out_file, method=3)


class PolynomHB(
    SlopeProvider
):  # 6 parameter 2nd order polynom (Heerdegen & Beran 1982)
    def get_slope(self, dem: StrPath, out_file: StrPath) -> StrPath:
        return compute_3x3_slope(dem, out_file, method=4)


class PolynomBRB(
    SlopeProvider
):  # 6 parameter 2nd order polynom (Bauer, Rohdenburg, Bork 1985)
    def get_slope(self, dem: StrPath, out_file: StrPath) -> StrPath:
        return compute_3x3_slope(dem, out_file, method=5)


class PolynomZT(
    SlopeProvider
):  # 9 parameter 2nd order polynom (Zevenbergen & Thorne 1987)
    def get_slope(self, dem: StrPath, out_file: StrPath) -> StrPath:
        return compute_3x3_slope(dem, out_file, method=6)


class PolynomH(SlopeProvider):  # 10 parameter 3rd order polynom (Haralick 1983)
    def get_slope(self, dem: StrPath, out_file: StrPath) -> StrPath:
        return compute_3x3_slope(dem, out_file, method=7)
