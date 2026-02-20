""" `utils.shapes`

Functions for calculating areas and volumes of common shapes.

Functions
---------
vol_cylinder(h, ro, ri)
    Volume of hollow right circular cylinder.
area_cylinder(h, r)
    Surface area of right circular cylinder.
vol_sphere(r)
    Volume of sphere.
area_sphere(r)
    Surface area of sphere.
area_circle(r)
    Area of circle.
hydraulic_diameter(area, perimeter)
    Calculates hydraulic diameter.
hydraulic_diameter_annulus(do, di)
    Calculates hydraulic diameter of an annulus.
"""
from __future__ import annotations
import numpy as np

# ----------------------------------------------------------------------------
# region: cylinders
# ----------------------------------------------------------------------------
def vol_cylinder(h, ro, ri=0.):
    """
    Calculates volume of right circular cylinder.

    Parameters
    ----------
    h : float or array-like
        Height.
    ro : float or array-like
        Outer radius.
    ri : float or array-like
        Inner radius.

    Returns
    -------
    float or ndarray
        Volume of hollow cylinder.
    """
    return np.pi * (ro ** 2 - ri ** 2) * h


def area_cylinder(h, r):
    """
    Calculates surface area of right circular cylinder.

    Parameters
    ----------
    h : float or array-like
        Height.
    r : float or array-like
        Radius.
        
    Returns
    -------
    float or ndarray
        Surface area of cylinder.
    """
    return np.pi * 2 * r * h


# ----------------------------------------------------------------------------
# region: spheres
# ----------------------------------------------------------------------------
def vol_sphere(r):
    """
    Calculates volume of sphere.

    Parameters
    ----------
    r : float or array-like
        Radius.
        
    Returns
    -------
    float or ndarray
        Volume of sphere.
    """
    return (4 / 3) * np.pi * r ** 3


def area_sphere(r):
    """
    Calculates surface area of sphere.

    Parameters
    ----------
    r : float or ndarray
        Radius.
        
    Returns
    -------
    float or ndarray
        Surface area of sphere.
    """
    return 4 * np.pi * r ** 2


# ----------------------------------------------------------------------------
# region: circles
# ----------------------------------------------------------------------------
def area_circle(ro, ri=0.):
    """
    Calculates area of circle.

    Parameters
    ----------
    ro : float or ndarray
        Outer radius.
    ri : float or ndarray
        Inner radius.

    Returns
    -------
    float or ndarray
        Area of circle.
    """
    return np.pi * (ro ** 2 - ri ** 2)


# ----------------------------------------------------------------------------
# region: hydraulic geometry
# ----------------------------------------------------------------------------
def hydraulic_diameter(area, perimeter):
    """
    Calculates hydraulic diameter.

    Parameters
    ----------
    area : float or ndarray
        Cross-sectional area.
    perimeter : float or ndarray
        Wetted perimeter.

    Returns
    -------
    float or ndarray
        Hydraulic diameter.
    """
    return 4 * area / perimeter


def hydraulic_diameter_annulus(do, di):
    """
    Calculates hydraulic diameter of an annulus.
    
    D_h = D_outer - D_inner

    Parameters
    ----------
    do : float or ndarray
        Outer diameter.
    di : float or ndarray
        Inner diameter.

    Returns
    -------
    float or ndarray
        Hydraulic diameter.
    """
    return do - di
