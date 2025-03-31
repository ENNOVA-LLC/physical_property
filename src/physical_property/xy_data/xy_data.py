""" `xy_data.xy_data`

For storage and visualization of (x,y) data at different series values (often time cuts).
"""
from attrs import define, field
from typing import Dict, List
import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger

from ..base import PhysicalProperty
from ..properties import Time, Length

logger.add("logs/xy_data.log", rotation="10 MB", level="DEBUG")

@define
class XYData:
    """
    Class to store and visualize (x,y) data at different series values.

    Attributes
    ----------
    time_cuts : Time
        A list of values at which data has been recorded.
    x_data : PhysicalProperty or Length
        The x values (e.g., location in the pipe) with name, unit, and value.
    properties_data : Dict[str, List[PhysicalProperty]]
        Keys are property names and values are lists of PhysicalProperty instances representing the y data at different time cuts.
    """
    time_cuts: Time = field()
    x_data: PhysicalProperty = field()
    properties_data: Dict[str, List[PhysicalProperty]] = field(factory=dict)

    def __init__(self, time_cuts, x_data):
        """
        Parameters
        ----------
        time_cuts : list, numpy array, or Time
            A list of time cuts at which data has been recorded. Assumed to be in seconds if not a Time instance.
        x_data : list, numpy array, or PhysicalProperty
            The x values (e.g., location in the pipe).
        """
        logger.info("Initializing XYData class.")
        # Convert time_cuts to Time instance if not already
        if not isinstance(time_cuts, PhysicalProperty):
            time_cuts = Time(name="time", unit="s", value=time_cuts)    # Assuming time in seconds
        # Convert x_data to PhysicalProperty if not already
        if not isinstance(x_data, PhysicalProperty):
            x_data = Length(name="length", unit=None, value=x_data)     # Assuming Length for spatial data
        
        object.__setattr__(self, "time_cuts", time_cuts)
        object.__setattr__(self, "x_data", x_data)
        object.__setattr__(self, "properties_data", {})

    # region DATA MANAGEMENT
    def add_data(self, property_name: str, y_data: PhysicalProperty) -> None:
        """
        Add data for a specific property at a specific time cut.

        Parameters
        ----------
        property_name : str
            The name of the property (e.g., density, viscosity).
        y_data : list, numpy array, or PhysicalProperty
            The y values (e.g., value of the property at corresponding x values).
        """
        if not isinstance(y_data, PhysicalProperty):
            y_data = PhysicalProperty(name=property_name, unit=None, value=np.asarray(y_data))
        
        if len(y_data.value) != len(self.x_data.value):
            raise ValueError(f"Inconsistent data dimensions for {property_name}.")
        
        if property_name not in self.properties_data:
            self.properties_data[property_name] = []
        
        self.properties_data[property_name].append(y_data)

    # region PLOTTING UTILITIES
    def plot(self, properties=None, downsample_factor=1, max_plots_per_figure=8, time_unit="day") -> list:
        """
        Generate (x, y) plots for the specified properties over all recorded time cuts using Plotly.

        Parameters
        ----------
        properties : list of str, optional
            The list of property names to plot. If None, all properties are plotted.
        downsample_factor : int, optional
            Factor to downsample the data. Default is 1 (not applied).
        max_plots_per_figure : int, optional
            Maximum number of plots per figure. Default is 8.
        time_unit : str, optional
            The unit for displaying time in the plot legend. Default is 'day'.

        Returns
        -------
        list of go.Figure
            A list of figures.
        """
        logger.info("Plot data in XYData class.")
        if properties is None:
            properties = list(self.properties_data.keys())

        if downsample_factor > 1:
            self._downsample_data(downsample_factor)

        figs = []
        for i in range(0, len(properties), max_plots_per_figure):
            figure_properties = properties[i:i + max_plots_per_figure]
            fig = self._create_subplot_figure(figure_properties, time_unit)
            figs.append(fig)

        return figs

    def _downsample_data(self, factor):
        """
        Downsample the x_data and y_data by the given factor.

        Parameters
        ----------
        factor : int
            The downsample factor.
        """
        self.x_data.value = self.x_data.value[::factor]
        for prop in self.properties_data:
            for y_data in self.properties_data[prop]:
                y_data.value = y_data.value[::factor]

    def _create_subplot_figure(self, properties, time_unit) -> go.Figure:
        """
        Private method to create a figure with subplots for multiple properties.

        Parameters
        ----------
        properties : list
            The properties to be plotted.
        time_unit : str
            The unit for displaying time in the plot legend.
        """
        logger.info("Creating subplots.")
        colors = [
            "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
            "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"
        ]

        rows = np.ceil(len(properties) / 2)
        fig = make_subplots(rows=int(rows), cols=2, shared_xaxes=False)

        x_label = f"{self.x_data.name} ({self.x_data.unit})" if self.x_data.unit else self.x_data.name

        for idx, prop in enumerate(properties):
            if prop not in self.properties_data:
                logger.warning(f"Property '{prop}' not found.")
                continue

            row = (idx // 2) + 1
            col = (idx % 2) + 1

            # Plot each time cut for this property
            y_min, y_max = float("inf"), float("-inf")
            for i, (time_cut, y_data) in enumerate(zip(self.time_cuts.value, self.properties_data[prop])):
                y_min = min(y_min, np.min(y_data.value))
                y_max = max(y_max, np.max(y_data.value))
                y_label = f"{y_data.name} ({y_data.unit})" if y_data.unit else y_data.name
                
                # Convert time to the specified unit for display
                if time_unit == "day":
                    time_display = time_cut / 86400
                    unit_str = "day"
                elif time_unit == "hour":
                    time_display = time_cut / 3600
                    unit_str = "hour"
                else:  # Default to seconds
                    time_display = time_cut
                    unit_str = "s"
                
                fig.add_trace(go.Scatter(
                    x=self.x_data.value, y=y_data.value, mode="lines", name=f"t={time_display:.2f} {unit_str}",
                    line=dict(color=colors[i % len(colors)]),
                    showlegend=(idx == 0)
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
            title={"text": "<b>Y(x) at each interval</b>", "font": {"family": "Arial", "size": 40}},
            height=400 * rows,
            width=1400,
            margin=dict(l=50, r=50, t=100, b=100),
            showlegend=True,
            plot_bgcolor="white"
        )
        return fig
    # endregion

if __name__ == "__main__":
    logger.info("Running xy_data module...")
