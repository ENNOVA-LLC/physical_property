""" `xy_data.xy_data`

For storage and visualization of (x,y) data across multiple series (e.g., time cuts or other indices).
"""
from attrs import define, field
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from plotly import graph_objects as go
from plotly.subplots import make_subplots

from ..base import PhysicalProperty
from ..properties import Time, Length

# Configure logging
from ..utils.logging import get_logger
logger = get_logger(__name__)

@define
class XYData:
    """
    Class to store and visualize (x,y) data across multiple series.
    
    The primary use case is for storing results of multiphase flow calculations at different time cuts or sample points, 
    where x could be location along the pipe and y could be various properties (pressure, velocity, etc.) at those locations.
    Each series corresponds to a different time cut, and the class can handle multiple properties for each series.

    Attributes
    ----------
    x_data : PhysicalProperty
        The x values (e.g., location in the pipe) with name, unit, and value.
    properties_data : Dict[str, List[PhysicalProperty]]
        Keys are property names and values are lists of PhysicalProperty instances representing the y data for each series.
    series_index : PhysicalProperty, optional
        Identifiers for each series (e.g., time cuts, sample numbers). If None, defaults to numeric indices (series-0, series-1, ...).
    """
    x_data: PhysicalProperty = field()
    properties_data: Dict[str, List[PhysicalProperty]] = field(factory=dict)
    series_index: Optional[PhysicalProperty] = field(default=None)

    def __init__(self, x_data, series_index=None):
        """
        Parameters
        ----------
        x_data : list, numpy array, or PhysicalProperty
            The x values (e.g., location in the pipe).
        series_index : list, numpy array, or PhysicalProperty, optional
            Identifiers for each series (e.g., time cuts). If None, defaults to numeric indices when y_data is added.
        """
        logger.debug("Initializing XYData class.")
        # Convert x_data to PhysicalProperty if not already
        if not isinstance(x_data, PhysicalProperty):
            x_data = Length(name="length", unit=None, value=x_data)  # Assuming Length for spatial data
        
        # Convert series_index to PhysicalProperty if provided and not already a PhysicalProperty
        if series_index is not None and not isinstance(series_index, PhysicalProperty):
            series_index = PhysicalProperty(name="series", unit=None, value=series_index)

        object.__setattr__(self, "series_index", series_index)
        object.__setattr__(self, "x_data", x_data)
        object.__setattr__(self, "properties_data", {})

    # ----------------------------------------------------------------------------
    # region: Data Management
    # ----------------------------------------------------------------------------
    def add_y_data(self, y_data: Union[PhysicalProperty, np.ndarray, list], y_name: str = None) -> None:
        """
        Add data for a specific property.

        Parameters
        ----------
        y_data : list, numpy array, or PhysicalProperty
            The y values (e.g., value of the property at corresponding x values).
        y_name : str, optional
            The name of the property. If not provided, it will be inferred from the y_data or default to "property".
        """
        if not isinstance(y_data, PhysicalProperty):
            y_name = y_name if y_name is not None else "property"
            if y_name in self.properties_data:
                y_name = f"{y_name}_{len(self.properties_data[y_name])}"
            y_data = PhysicalProperty(name=y_name, unit=None, value=np.asarray(y_data))
        else:
            y_name = y_data.name

        self._validate_and_store_y_data(y_data, y_name)
        
        # Update series_index if not provided
        if self.series_index is None:
            num_series = len(self.properties_data[y_name])
            object.__setattr__(self, "series_index", PhysicalProperty(
                name="series", 
                unit=None, 
                value=np.arange(num_series)
            ))
        elif len(self.properties_data[y_name]) > len(self.series_index.value):
            logger.warning(
                "Number of y_data entries (%s) exceeds series_index (%s) for %s",
                len(self.properties_data[y_name]),
                len(self.series_index.value),
                y_name,
            )

    def add_series_data(self, series_index_val: Union[float, int, str], y_data_dict: Dict[str, Union[PhysicalProperty, np.ndarray, list]]) -> None:
        """
        Add a collection of y(x) data for a new series index value.

        Parameters
        ----------
        series_index_val : float, int, or str
            The new series index value (e.g., a time cut like 3.0 or "sample_1").
        y_data_dict : dict
            A dictionary mapping property names to their y values (e.g., {"pressure": [1, 2, 3], "temp": [300, 310, 320]}).
            Values can be lists, numpy arrays, or PhysicalProperty instances.
        """
        logger.debug("Adding series data for series_index_val=%s", series_index_val)

        # Update series_index
        if self.series_index is None:
            object.__setattr__(self, "series_index", PhysicalProperty(
                name="series", 
                unit=None, 
                value=[series_index_val]
            ))
        else:
            new_series = np.append(self.series_index.value, series_index_val)
            object.__setattr__(self, "series_index", PhysicalProperty(
                name=self.series_index.name, 
                unit=self.series_index.unit, 
                value=new_series
            ))

        # Add each property's y_data
        for y_name, y_data in y_data_dict.items():
            if not isinstance(y_data, PhysicalProperty):
                y_data = PhysicalProperty(name=y_name, unit=None, value=np.asarray(y_data))
            self._validate_and_store_y_data(y_data, y_name)

    def _validate_and_store_y_data(self, y_data, y_name):
        """Utility to validate and store y_data."""
        if len(y_data.value) != len(self.x_data.value):
            raise ValueError(
                f"Inconsistent data dimensions for {y_name}: x_data has {len(self.x_data.value)} points, y_data has {len(y_data.value)} points"
            )
        if y_name not in self.properties_data:
            self.properties_data[y_name] = []
        self.properties_data[y_name].append(y_data)

    # ----------------------------------------------------------------------------
    # region: Plotting Utils
    # ----------------------------------------------------------------------------
    def plot(self, properties=None, downsample_factor=1, max_plots_per_figure=10) -> List[go.Figure]:
        """
        Generate (x, y) plots for the specified properties, with each plot showing all series.

        Parameters
        ----------
        properties : list of str, optional
            The list of property names to plot. If None, all properties are plotted.
        downsample_factor : int, optional
            Factor to downsample the x and y data (series_index remains unchanged). Default is 1 (not applied).
        max_plots_per_figure : int, optional
            Maximum number of plots per figure. Default is 10.

        Returns
        -------
        list of go.Figure
            A list of figures.
        """
        logger.debug("Plotting data in XYData class.")
        if not self.properties_data:
            raise ValueError("No y-data available to plot.")
        
        if properties is None:
            properties = list(self.properties_data.keys())

        if downsample_factor > 1:
            self._downsample_data(downsample_factor)

        figs = []
        for i in range(0, len(properties), max_plots_per_figure):
            figure_properties = properties[i:i + max_plots_per_figure]
            fig = self._create_subplot_figure(figure_properties)
            figs.append(fig)

        return figs

    def _downsample_data(self, factor):
        """
        Downsample the x_data and y_data by the given factor (series_index is not downsampled).

        Parameters
        ----------
        factor : int
            The downsample factor.
        """
        self.x_data.update_value(self.x_data.value[::factor])
        for prop in self.properties_data:
            for y_data in self.properties_data[prop]:
                y_data.update_value(y_data.value[::factor])

    def _create_subplot_figure(self, properties: List[str]) -> go.Figure:
        """
        Private method to create a figure with subplots for multiple properties, each showing all series.

        Parameters
        ----------
        properties : List[str]
            The properties to be plotted.
        """
        logger.debug("Creating subplots.")
        colors = [
            "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
            "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"
        ]

        rows = np.ceil(len(properties) / 2)
        fig = make_subplots(rows=int(rows), cols=2, shared_xaxes=False)

        x_label = f"{self.x_data.name} ({self.x_data.unit})" if self.x_data.unit else self.x_data.name
        y_title = "Y" # ', '.join(properties)
        title_text = f"<b>{y_title} v. {x_label}</b>"

        for idx, prop in enumerate(properties):
            if prop not in self.properties_data:
                logger.warning("Property '%s' not found.", prop)
                continue

            row = (idx // 2) + 1
            col = (idx % 2) + 1

            # Plot each series for this property
            y_min, y_max = float("inf"), float("-inf")
            y_data_list = self.properties_data[prop]
            
            for i, y_data in enumerate(y_data_list):
                y_min = min(y_min, np.min(y_data.value))
                y_max = max(y_max, np.max(y_data.value))
                y_label = f"{y_data.name} ({y_data.unit})" if y_data.unit else y_data.name
                
                # Use series_index for legend if available, otherwise use index
                if self.series_index is not None and i < len(self.series_index.value):
                    series_val = self.series_index.value[i]
                    series_unit = f" ({self.series_index.unit})" if self.series_index.unit else ""
                    name = f"{self.series_index.name}={series_val:.2f}{series_unit}"
                else:
                    name = f"{y_data.name}_{i}"
                
                # Show legend only for the first subplot (idx == 0)
                show_legend = (idx == 0)
                
                fig.add_trace(go.Scatter(
                    x=self.x_data.value, 
                    y=y_data.value, 
                    mode="lines", 
                    name=name,
                    line=dict(color=colors[i % len(colors)]),
                    showlegend=show_legend
                ), row=row, col=col)

            # Set y-axis limits
            y_range = y_max - y_min
            if y_range < 1e-3:
                margin = max(1e-3, y_max * 0.01)
                y_min -= margin
                y_max += margin

            # Customize layout for each subplot
            axes_dict = dict(row=row, col=col, showgrid=True, gridcolor="#e5e8e8", zerolinecolor="#e5e8e8", zerolinewidth=1, linecolor="black", mirror=True, ticks="outside")
            fig.update_xaxes(
                title={"text": f"<b>{x_label}</b>"},
                **axes_dict
            )
            fig.update_yaxes(
                title={"text": f"<b>{y_label}</b>", "font": {"family": "Arial", "size": 23}},
                range=[y_min, y_max],
                **axes_dict
            )

        fig.update_layout(
            title={"text": title_text, "font": {"family": "Arial", "size": 40}},
            height=400 * rows,
            width=1400,
            margin=dict(l=50, r=50, t=100, b=100),
            showlegend=True,
            plot_bgcolor="white"
        )
        return fig
    # endregion

    # ----------------------------------------------------------------------------
    # region: Magic Methods
    # ----------------------------------------------------------------------------
    def __str__(self):
        """
        Return a string representation of the XYData instance, including the x_data, properties_data, and series_index.

        Returns
        -------
        str
            A string representation of the XYData instance.
        """
        series_str = f"series_index={self.series_index.value}" if self.series_index is not None else "series_index=None"
        x_str = f"x_data={self.x_data.name} ({len(self.x_data.value)} points, unit={self.x_data.unit})"
        props_str = ", ".join([f"{key} ({len(val)} series)" for key, val in self.properties_data.items()])
        return f"XYData(\n  {series_str}, \n  {x_str}, \n  properties=[{props_str}])"
    # endregion

    # ----------------------------------------------------------------------------
    # region: Dataframes
    # ----------------------------------------------------------------------------
    def to_frames_by_series(self, properties: Optional[List[str]] = None, include_units: bool = True) -> Dict[Union[int, float, str], pd.DataFrame]:
        """
        Convert the XYData instance to a dictionary of pandas DataFrames, where each DataFrame represents a series of data.

        Parameters
        ----------
        properties : List[str], optional
            List of property names to include. If None, all properties are included.
        include_units : bool, optional
            Whether to include units in the column names. Defaults to True.

        Returns
        -------
        Dict[Union[int, float, str], pd.DataFrame]
            A dictionary mapping series indices (values from series_index if available, else 0-based integers)
            to their corresponding DataFrames.
        """
        if not self.properties_data:
            return {}

        props = properties or list(self.properties_data.keys())

        # helper for "Name [unit]"
        def col_name(pp: PhysicalProperty) -> str:
            if not include_units or not getattr(pp, "unit", None):
                return pp.name
            return f"{pp.name} [{pp.unit}]"

        # pre-build X column
        x_col_name = col_name(self.x_data)
        x_vals = self.x_data.value

        # how many series? (length of lists under any property)
        n_series = len(next(iter(self.properties_data.values())))
        
        # Determine keys for the dictionary
        keys = list(range(n_series))
        if self.series_index is not None and len(self.series_index.value) >= n_series:
            # use the actual values from series_index
            keys = self.series_index.value[:n_series]
            # Convert numpy scalars to python types for better dict compatibility
            keys = [k.item() if hasattr(k, "item") else k for k in keys]

        frames: Dict[Union[int, float, str], pd.DataFrame] = {}

        for i, key in enumerate(keys):
            cols = {x_col_name: x_vals}
            for p in props:
                ypp = self.properties_data.get(p, [])
                if i < len(ypp):
                    cols[col_name(ypp[i])] = ypp[i].value
            frames[key] = pd.DataFrame(cols)

        return frames

    def to_df(self, series_index_val: Union[int, float, str], properties: Optional[List[str]] = None, include_units: bool = True) -> pd.DataFrame:
        """
        Get a table (DataFrame) of the XY data for a specific series.

        Parameters
        ----------
        series_index_val : int, float, or str
            The value of the series index to retrieve. If series_index is not set on the object,
            this should be the 0-based integer index of the series.
        properties : List[str], optional
            List of property names to include. If None, all properties are included.
        include_units : bool, optional
            Whether to include units in the column names. Defaults to True.

        Returns
        -------
        pd.DataFrame
            DataFrame containing x_data and the specified properties for the given series.

        Raises
        ------
        ValueError
            If the specified series_index_val is not found.
        """
        frames = self.to_frames_by_series(properties=properties, include_units=include_units)
        
        if series_index_val in frames:
            return frames[series_index_val]
        
        # Fallback: exact match might fail due to float precision or type (int vs float)
        # Try to find a match
        for key in frames.keys():
            if key == series_index_val: # This handles int/float comparison (1 == 1.0)
                return frames[key]
            try:
                if np.isclose(float(key), float(series_index_val)):
                    return frames[key]
            except (ValueError, TypeError):
                pass
                
        available = list(frames.keys())
        raise ValueError(f"Series index '{series_index_val}' not found. Available series: {available}")
    # endregion

# ----------------------------------------------------------------------------
# region: Example Usage
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Running xy_data module...")
    # Example: Single series
    x = np.linspace(0, 10, 100)
    data = XYData(x_data=Length(name="Pipe Length", unit="m", value=x))
    data.add_series_data(series_index_val=0, y_data_dict={
        "pressure": np.sin(x),
        "temperature": np.cos(x)
    })
    figs = data.plot()
    figs[0].show()

    # Example: Multiple series
    data = XYData(x_data=Length(name="Pipe Length", unit="m", value=x))
    data.add_series_data(series_index_val=0, y_data_dict={
        "pressure": np.sin(x),
        "temperature": np.cos(x)
    })
    data.add_series_data(series_index_val=1, y_data_dict={
        "pressure": np.sin(x + 1),
        "temperature": np.cos(x + 1)
    })
    data.add_series_data(series_index_val=2, y_data_dict={
        "pressure": np.sin(x + 2),
        "temperature": np.cos(x + 2)
    })
    figs = data.plot()
    figs[0].show()
    