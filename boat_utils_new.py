__all__ = [
    "make_hull",
    "hull_data",
    "hull_rotate",
    "get_avs",
    "get_mass_properties",
    "get_moment_curve",
    "get_buoyant_properties",
    "get_equ_waterline",
    "RHO_WATER",
    "fun_avs",
    "fun_moment",
]

from numpy import array, average, concatenate, linspace, meshgrid, sum, min
from numpy import abs, sin, cos, pi, NaN
from pandas import concat, DataFrame
from scipy.optimize import bisect
from warnings import catch_warnings, simplefilter

import numpy as np
import grama as gr
DF = gr.Intention()


# Global constants
RHO_WATER = 0.03613 # Density of water (lb / in^3)
RHO_0 = 0.04516 # Filament density (lb / in^3)
G = 386 # Gravitational acceleration (in / s^2)

## Boat generator
# --------------------------------------------------
def make_hull(X):
    r"""

    Args:
        X (iterable): [H, W, n, d] = X

        H = height of boat [in]
        W = width of boat [in]
        n = shape parameter [-]
        d = displacement ratio [-]
          = weight of boat / weight of max displaced water

    Returns:
        DataFrame: Hull points
        DataFrame: Mass properties
    """
    H, W, n, d = X

    f_hull = lambda x: H * abs(2 * x / W)**n
    g_top = lambda x, y: y <= H
#     rho_hull = lambda x, y: RHO_0 * (y <= h_k) + 0.25 * RHO_0 * (y > h_k)
    rho_hull = lambda x, y: d * RHO_WATER

    df_hull, dx, dy = hull_data(
        f_hull,
        g_top,
        rho_hull,
        n_marg=100,
    )

    df_mass = get_mass_properties(df_hull, dx, dy)

    return df_hull, df_mass

## Hull manipulation
# --------------------------------------------------
def hull_data(f_hull, g_top, rho_hull, n_marg=50, x_wid=3, y_lo=+0, y_hi=+4):
    r"""
    Args:
        f_hull (lambda): Function of signature y = f(x);
            defines lower surface of boat
        g_top (lambda): Function of signature g (bool) = g(x, y);
            True indicates within boat
        rho_hull (lambda): Function of signature rho = rho(x, y);
            returns local hull density

    Returns:
        DataFrame: x, y, dm boat hull points and element masses
        float: dx
        float: dy
    """
    Xv = linspace(-x_wid, +x_wid, num=n_marg)
    Yv = linspace(y_lo, y_hi, num=n_marg)
    dx = Xv[1] - Xv[0]
    dy = Yv[1] - Yv[0]

    Xm, Ym = meshgrid(Xv, Yv)
    n_tot = Xm.shape[0] * Xm.shape[1]

    Z = concatenate(
        (Xm.reshape(n_tot, -1), Ym.reshape(n_tot, -1)),
        axis=1,
    )

    M = array([rho_hull(x, y) * dx * dy for x, y in Z])

    I_hull = [
        (f_hull(x) <= y) & g_top(x, y)
        for x, y in Z
    ]
    Z_hull = Z[I_hull]
    M_hull = M[I_hull]

    df_hull = DataFrame(dict(
        x=Z_hull[:, 0],
        y=Z_hull[:, 1],
        dm=M_hull,
    ))

    return df_hull, dx, dy

def hull_rotate(df_hull, df_mass, angle):
    r"""
    Args:
        df_hull (DataFrame): Hull points
        df_mass (DataFrame): Mass properties, gives COM
        angle (float, radians): Heel angle

    Returns:
        DataFrame: Hull points rotated about COM
    """
    R = array([
        [cos(angle), -sin(angle)],
        [sin(angle),  cos(angle)]
    ])
    Z_hull_r = (
        df_hull[["x", "y"]].values - df_mass[["x", "y"]].values
    ).dot(R.T) + df_mass[["x", "y"]].values

    return DataFrame(dict(
        x=Z_hull_r[:, 0],
        y=Z_hull_r[:, 1],
        dm=df_hull.dm,
    ))

## Evaluate hull
# --------------------------------------------------
def get_mass_properties(df_hull, dx, dy):
    x_com = average(df_hull.x, weights=df_hull.dm)
    y_com = average(df_hull.y, weights=df_hull.dm)
    mass = df_hull.dm.sum()

    return DataFrame(dict(
        x=[x_com],
        y=[y_com],
        dx=[dx],
        dy=[dy],
        mass=[mass]
    ))

def get_buoyant_properties(df_hull_rot, df_mass, w_slope, w_intercept):
    r"""
    Args:
        df_hull_rot (DataFrame): Rotated hull points
        df_mass (DataFrame): Mass properties
        eq_water (lambda): takes in an x value, spits out a y
    """
# x and y intervals
    dx = df_mass.dx[0]
    dy = df_mass.dy[0]

# Define equation for the surface of the water using slope intercept form
    eq_water = lambda x: w_slope * x + w_intercept

# Find points under sloped waterline
    df_hull_under = (
        df_hull_rot
        >> gr.tf_mutate(
            under = df_hull_rot.y <= eq_water(df_hull_rot.x)
        )
        >> gr.tf_filter(
            DF.under == True
        )
    )

# x and y position of COB
    x_cob = average(df_hull_under.x)
    y_cob = average(df_hull_under.y)

# Pull x and y of COM as well
    x_com = df_mass.x[0]
    y_com = df_mass.y[0]

# Total mass of water by finding area under curve
    m_water = RHO_WATER * len(df_hull_under) * dx * dy

# Net force results from the difference in masses between the boat and the water
    F_net = (m_water - df_mass.mass[0]) * G

# Distance to determine torque is ORTHOGONAL TO WATERLINE
# Equation from https://www.geeksforgeeks.org/perpendicular-distance-between-a-point-and-a-line-in-2-d/

# Calculate righting moment for different cases of w_slope
# Account for zero slope
    if w_slope == 0:
        M_net = G * m_water * x_cob
    else:
#         norm_dist = ((1 * y_com) + (1/w_slope * x_com) + ((1/w_slope) * x_cob) + y_cob) / np.sqrt(1 + (1/w_slope)**2)
        a = 1/w_slope
        b = 1
        c = (1/w_slope) * x_cob + y_cob
        x1 = x_com
        y1 = y_com

        norm_dist = (a*x1 + b*y1 + c) / np.sqrt(a**2 + b**2)

        if w_slope > 0: # positive water slope creates a positive moment
            M_net = G * m_water * norm_dist
        elif w_slope < 0: # negative water slope creates a negative moment
            M_net = - G * m_water * norm_dist


    return DataFrame(dict(
        x=[x_cob],
        y=[y_cob],
        F_net=[F_net],
        M_net=[M_net],
    ))

def get_equ_waterline(df_hull, df_mass, boat_angle, water_angle, y_l=0, y_h=4):
    r"""
    Args:
        df_hull (DataFrame): Unrotated hull points
        df_mass (DataFrame): Mass properties
        boat_angle (float): Angle of hull rotation
        water_angle (float): Angle of waterline
        y_l (float): Low-bound for waterline
        y_h (float): High-bound for waterline

    Returns:
        float: Waterline of zero net vertical force (heave-steady)
    """
# position of COM
    dx = df_mass.dx[0]
    dy = df_mass.dy[0]

# Rotate the hull relative to a steady waterline
    df_hull_r = hull_rotate(df_hull, df_mass, boat_angle)

# Turn angle into a slope
    m = np.tan(water_angle)

# Make a function which calls get_buoyant properties at different values of y_g
# -> vertical offset for the waterline which is no longer level necessarily
    def fun(y_g):
        df_buoy = get_buoyant_properties(
            df_hull_r,
            df_mass,
            m,
            y_g,
        )
# All that matters is the buoyant force! because we are interested in seeing what waterline floats the boat
        return df_buoy.F_net[0]

    try:
        with catch_warnings():
            simplefilter("ignore")
# Important that the buoyant force for one input is positive while the other is negative for scipy bisect
            y_star = bisect(fun, y_l, y_h, maxiter=1000)


        df_res = get_buoyant_properties(
                df_hull_r,
                df_mass,
                m,
                y_star,
            )
        df_res["y_w"] = y_star
    except ValueError:
        df_res = DataFrame(dict(M_net=[NaN], y_w=[NaN]))

    return df_res

def get_moment_curve(df_hull, df_mass, water_angle, a_l=0, a_h=np.pi, num=50):
    r"""Generate a righting moment curve

    Args:
        df_hull (DataFrame): Unrotated hull points
        df_mass (DataFrame): Mass properties
        a_l (float): Low-bound for angle
        a_h (float): High-bound for angle
        num (int): Number of points to sample (linearly) between a_l, a_h

    Returns:
        DataFrame: Data from angle sweep
    """
    df_res = DataFrame()
    a_all = linspace(a_l, a_h, num=num)

    for boat_angle in a_all:
        df_tmp = get_equ_waterline(df_hull, df_mass, boat_angle, water_angle)
        df_tmp["boat_angle"] = boat_angle

        df_res = concat((df_res, df_tmp), axis=0)
    df_res.reset_index(inplace=True, drop=True)

    return df_res




# def get_buoyant_properties(df_hull_rot, df_mass, w_slope, w_intercept):
#     r"""
#     Args:
#         df_hull_rot (DataFrame): Rotated hull points
#         df_mass (DataFrame): Mass properties
#         eq_water (lambda): takes in an x value, spits out a y
#     """
# # x and y intervals
#     dx = df_mass.dx[0]
#     dy = df_mass.dy[0]

# # I_under is the boolean array of points under y_water


# # Define equation for the surface of the water using slope intercept form
#     eq_water = lambda x: w_slope * x + w_intercept

# # Find points under sloped waterline
#     df_hull_under = (
#         df_hull_rot
#         >> gr.tf_mutate(
#             test = eq_water(df_hull_rot.x),
#             under = df_hull_rot.y <= eq_water(df_hull_rot.x)
#         )
#         >> gr.tf_filter(
#             DF.under == True
#         )
#     )

# # x and y position of COB
#     x_cob = average(df_hull_under.x)
#     y_cob = average(df_hull_under.y)

# # Pull x and y of COM as well
#     x_com = df_mass.x[0]
#     y_com = df_mass.y[0]

# # Total mass of water by finding area under curve
#     m_water = RHO_WATER * len(df_hull_under) * dx * dy

# # Force of buoyancy is mass of water times gravitational constant
#     F_net = (m_water - df_mass.mass[0]) * G

# # Distance to determine torque is ORTHOGONAL TO WATERLINE
# # Equation from https://www.geeksforgeeks.org/perpendicular-distance-between-a-point-and-a-line-in-2-d/

# # Account for zero slope
#     try:
#         norm_dist = (1 + (1/w_slope * x_com) + (1/w_slope * x_cob) + y_cob) / np.sqrt(1 + (1/w_slope)**2)
#         # if w_slope > 0:
#         #     norm_dist = -norm_dist
#         M_net = G * m_water * norm_dist
#     except:
#         M_net = 0




#     return DataFrame(dict(
#         x=[x_cob],
#         y=[y_cob],
#         F_net=[F_net],
#         M_net=[M_net],
#     ))

# def get_equ_waterline(df_hull, df_mass, boat_angle, water_angle, y_l=0, y_h=4):
#     r"""
#     Args:
#         df_hull (DataFrame): Unrotated hull points
#         df_mass (DataFrame): Mass properties
#         angle (float): Angle of rotation
#         y_l (float): Low-bound for waterline
#         y_h (float): High-bound for waterline

#     Returns:
#         float: Waterline of zero net vertical force (heave-steady)
#     """
# # position of COM
#     dx = df_mass.dx[0]
#     dy = df_mass.dy[0]

# # Rotate the hull relative to a steady waterline
#     df_hull_r = hull_rotate(df_hull, df_mass, boat_angle)

# # Turn angle into a slope
#     m = np.tan(water_angle)

# # Make a function which calls get_buoyant properties at different values of y_g
# # -> might want to make this just a vertical offset for the waterline which is not level
#     def fun(y_g):
#         df_buoy = get_buoyant_properties(
#             df_hull_r,
#             df_mass,
#             m,
#             y_g,
#         )
# # All that matters is the buoyant force! because we are interested in seeing what waterline floats the boat
#         return df_buoy.F_net[0]

#     try:
#         with catch_warnings():
#             simplefilter("ignore")
# # Important that the buoyant force for one input is positive while the other is negative for scipy bisect
#             y_star = bisect(fun, y_l, y_h, maxiter=1000)


#         df_res = get_buoyant_properties(
#                 df_hull_r,
#                 df_mass,
#                 m,
#                 y_star,
#             )
#         df_res["y_w"] = y_star
#     except ValueError:
#         df_res = DataFrame(dict(M_net=[NaN], y_w=[NaN]))

#     return df_res

# def get_moment_curve(df_hull, df_mass, water_angle, a_l=0, a_h=np.pi, num=50):
#     r"""Generate a righting moment curve

#     Args:
#         df_hull (DataFrame): Unrotated hull points
#         df_mass (DataFrame): Mass properties
#         a_l (float): Low-bound for angle
#         a_h (float): High-bound for angle
#         num (int): Number of points to sample (linearly) between a_l, a_h

#     Returns:
#         DataFrame: Data from angle sweep
#     """
#     df_res = DataFrame()
#     a_all = linspace(a_l, a_h, num=num)

#     for boat_angle in a_all:
#         df_tmp = get_equ_waterline(df_hull, df_mass, boat_angle, water_angle)
#         df_tmp["boat_angle"] = boat_angle

#         df_res = concat((df_res, df_tmp), axis=0)
#     df_res.reset_index(inplace=True, drop=True)

#     return df_res

def get_avs(df_hull, df_mass, a_l=0.1, a_h=pi - 0.1):
    r"""
    Args:
        df_hull (DataFrame): Unrotated hull points
        df_mass (DataFrame): Mass properties
        a_l (float): Low-bound for angle
        a_h (float): High-bound for angle

    Returns:
        float: Angle of vanishing stability
    """
    # Create helper function
    def fun(angle):
        df_res = get_equ_waterline(
            df_hull,
            df_mass,
            angle,
        )

        return df_res.M_net[0]

    # Bisect for zero-moment
    try:
        a_star = bisect(fun, a_l, a_h, maxiter=1000)

        df_res = get_equ_waterline(
            df_hull,
            df_mass,
            a_star,
        )
        df_res["angle"] = a_star
    except ValueError:
        df_res = DataFrame(dict(angle=[NaN]))

    return df_res

## Convenience Functions
# --------------------------------------------------
def fun_avs(X, num=15):
    r"""Compute AVS of boat design, stably

    For numerical stability, find a lower bracket for the AVS then run bisection.

    Args:
        X (iterable): [H, W, n, d] = X

        H (float): height of boat [in]
        W (float): width of boat [in]
        n (float): shape parameter [-]
        d (float): displacement ratio [-]
                   weight of boat / weight of max displaced water
        num (int): number of points for moment sweep; should be coarse (num < ~20)

    Returns:
        float: Angle of vanishing stability
    """
    ## Generate boat hull given parameters
    df_hull, df_mass = make_hull(X)

    ## Generate coarse moment curve to find a bracket for AVS
    df_moment = get_moment_curve(df_hull, df_mass, num=num)

    ## Find lowest bracket for root (AVS)
    ind_h = next(i for i, df in df_moment[1:].iterrows() if df.M_net > 0)
    a_h = df_moment.iloc[ind_h].angle
    a_l = df_moment.iloc[ind_h - 1].angle

    ## Use the bracket to find AVS
    df_avs = get_avs(df_hull, df_mass, a_l=a_l, a_h=a_h)

    return df_avs.angle[0]

def fun_moment(X, num=50):
    r"""Compute net moment curve

    Args:
        X (iterable): [H, W, n, d] = X

        H (float): height of boat [in]
        W (float): width of boat [in]
        n (float): shape parameter [-]
        d (float): displacement ratio [-]
                   weight of boat / weight of max displaced water
        num (int): number of points for moment sweep

    Returns:
        df: Moment curve
    """
    ## Generate boat hull given parameters
    df_hull, df_mass = make_hull(X)

    ## Generate coarse moment curve to find a bracket for AVS
    df_moment = get_moment_curve(df_hull, df_mass, num=num)

    return df_moment
