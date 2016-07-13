# Main dependencies
import matplotlib.pyplot as plt
import numpy as np

# Parent Regression plotting class
from .regression import RegressionPlotting

class NonlinearPlotting(RegressionPlotting):
    """Plotting object to attach to nonlinear epistasis model.

    Parameters
    ----------
    model : NonlinearEpistasisModel

    Attributes
    ----------
    """
    def __init__(self, model):
        self.model = model
        super(NonlinearPlotting, self).__init__(self.model)

    def linear_phenotypes(self):
        """ P vs. p plot. """
        fig, ax = plt.subplots()

        known = self.model.phenotypes
        predicted = np.dot(self.model.X,  self.model.epistasis.values)


        # Add scatter plot points on correlation grid
        ax.plot(predicted, known, 'b.')

        ax.set_xlabel("linear phenotypes")
        ax.set_ylabel("nonlinear phenotypes")

        return fig, ax

    def nonlinear_function(self, ax=None, xbounds=None, figsize=(6,4), **kwargs):
        """ Plot the input function for set of phenotypes. """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        params = self.model.Parameters._param_list

        # Get the values
        values = [getattr(self.model.Parameters, p) for p in params]

        if xbounds is None:
            predicted = np.dot(self.model.X,  self.model.Interactions.values)

            if self.model.Linear.log_transform:
                predicted = 10**predicted

            max_p = max(predicted)
            min_p = min(predicted)

        else:
            max_p = xbounds[1]
            min_p = xbounds[0]

        x = np.linspace(min_p, max_p, 1000)
        y = self.model.function(x, *values)
        ax.plot(x,y, **kwargs)

        return fig, ax


    def best_fit(self, ax=None, figsize=(6,4), errorbars=False, axis=None, **kwargs):
        """ Plot model line through data.

        Parameters:
        ----------
        ax : matplotlib.Axis
            matplotlib object to add best fit data to.

        """
        # Add to axis if given, else create new plot.
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Plot line through data
        linear = self.model.statistics.linear()
        theory = self.model.statistics.predict()
        observed = self.model.phenotypes

        data = np.array((linear, theory))
        data = data[:, data[0, :].argsort()]

        # Plot the line through data

        if errorbars is True:
                if self.model.Linear.log_transform:
                    upper = np.log10(1 + self.model.stdeviations/self.model.phenotypes)
                    lower = np.log10(1 - self.model.stdeviations/self.model.phenotypes)

                else:
                    upper = self.model.stdeviations
                    lower = upper

                ax.errorbar(linear, observed, yerr=[upper,abs(lower)], **kwargs)
        else:
            ax.plot(linear, observed, '.', **kwargs)

        #ax.plot(data[0], data[1], color="r", linewidth=2)
        fig, ax = self.nonlinear_function(ax=ax, color="r", linewidth=2)

        if axis is not None:
            ax.axis(axis)

        return fig, ax

    def residuals(self, ax=None, figsize=(6,4), stem=False, axis=None, **kwargs):
        """ Get figure, return figure. """

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()


        theory = self.model.statistics.predict()
        linear = self.model.statistics.linear()
        observed = self.model.phenotypes

        # Calculate residuals
        residuals =  theory - observed

        # Build array of residuals next to theoretical value
        # then sort those columns in ascending order of theoretical
        # value for plotting.
        data = np.array((linear, residuals))
        data = data[:, data[0, :].argsort()]

        ylim = max([abs(min(residuals)), abs(max(residuals))])

        # Create a stem plot of the data
        if stem:
            markerline, stemlines, baseline = ax.stem(data[0], data[1], markerfmt=" ", linewidth=6, color='b')
            plt.setp(markerline, 'markerfacecolor', 'b')
            plt.setp(stemlines, 'linewidth', 1.5)
            plt.setp(baseline, 'color','r', 'linewidth', 1)
        else:
            ax.errorbar(data[0], data[1], **kwargs)
            ax.hlines(0, min(data[0]), max(data[0]), linestyle="dotted")
        ax.set_ylim([-ylim, ylim])

        # Set axis if given
        if axis is not None:
            ax.axis(axis)

        return fig, ax
