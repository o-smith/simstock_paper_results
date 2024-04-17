"""
Module containing plotting routines.
"""

from functools import singledispatch
from typing import Any, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
from matplotlib.axes._axes import Axes
import contextily as cx
from shapely.wkt import loads
from shapely.geometry import (
    Polygon,
    Point,
    LineString
)
import pandas as pd


def plot_region(df: pd.DataFrame,
                 background: bool = True,
                 colour_variable: Union[str, None] = None,
                 colourmap: str = "viridis",
                 linewideth: float = 2.0,
                 ax: Axes = None
                 ) -> None:

    # Set up the plot
    if not ax:
        _, ax = plt.subplots(1, 1)

    if colour_variable:
        # Iterate over the edge data to get a list of
        # the values by which we want to colour edges
        colour_vals = []
        for _, row in df.iterrows():

            # Get the requested value from the edge if it is exists
            try:
                colour_vals.append(row[colour_variable])
            except KeyError as e:
                raise KeyError(f"{colour_variable} is not a known edge attribute.") from e
            
            norm = plt.Normalize(min(colour_vals), max(colour_vals))
            cmap = plt.cm.get_cmap(colourmap)
            scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            scalar_mappable.set_array([])


    # First, plot all edges
    for _, row in df.iterrows():

        if colour_variable:
            plot_geometry(loads(row["geometry"]), ax, color=scalar_mappable.to_rgba(row[colour_variable]), lw=linewideth)
        else:
            plot_geometry(loads(row["geometry"]), ax, color="red", alpha=0.6)


    # Add an open street map background if requested
    if background:
        cx.add_basemap(ax, alpha=0.6, crs="epsg:27700", source=cx.providers.OpenStreetMap.Mapnik)

    plt.xticks([])
    plt.yticks([])
    if colour_variable:
        plt.colorbar(scalar_mappable, label=colour_variable, ax=ax) 


@singledispatch
def plot_geometry(geom: Any, _: Axes):
    """
    Function to plot an individual shapely geometry.
    Each type of geometry (Point, Polygon, LineString)
    needs to be plotted differently. To handle this, this
    function is overloaded using singledispatch to accept
    each of these three types.
    """
    # If geom is not of a type for which an overloaded
    # function definition is given, then this
    # code will be reached, giving a type error.
    msg = f"Type: {type(geom)} cannot be used with function plot_geometry()"
    raise TypeError(msg)


@plot_geometry.register
def _(geom: Polygon, ax: Axes, **kwargs) -> PatchCollection:
    """
    Overloaded Polygon implementation of the
    plot_geometry function.
    """
    path = Path.make_compound_path(
        Path(np.asarray(geom.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in geom.interiors])
    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)
    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection

@plot_geometry.register
def _(geom: Point, ax: Axes, **kwargs) -> None:
    """
    Overloaded Point implementation of the
    plot_geometry function.
    """
    ax.scatter(geom.x, geom.y, **kwargs)

@plot_geometry.register
def _(geom: LineString, ax: Axes, **kwargs) -> None:
    """
    Overloaded LineString implementation of the
    plot_geometry function.
    """
    ax.plot(*geom.xy, **kwargs)
