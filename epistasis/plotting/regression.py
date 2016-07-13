# Main dependencies
import matplotlib.pyplot as plt
import numpy as np

# Parent Regression plotting class
from .models import EpistasisPlotting

class RegressionPlotting(EpistasisPlotting):

    def __init__(self, model):
        """ Reference by model or gpm."""
        self.model = model
        super(RegressionPlotting, self).__init__(self.model)

    def correlation(self, ax=None, figsize=(6,4), **kwargs):
        """ Draw a correlation plot of data. """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        max_p = max(self.model.phenotypes)
        min_p = min(self.model.phenotypes)

        known = self.model.phenotypes
        predicted = self.model.statistics.predict()

        # Add scatter plot points on correlation grid
        ax.plot(known, predicted, 'b.')

        # Add 1:1 correlation line
        ax.plot(np.linspace(min,max, 10), np.linspace(min,max, 10), 'r-')

        ax.set_xlabel("known")
        ax.set_ylabel("learned")

        return fig, ax

    def predicted_phenotypes(self, ax=None, figsize=(6,4), **kwargs):
        """
            Plots the predicted phenotypes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        known = self.model.phenotypes
        predicted = self.model.statistics.predict()

        # Add scatter plot points on correlation grid
        ax.plot(known, 'b-')
        ax.plot(predicted, 'r-')

        ax.set_ylabel("phenotypes")
        ax.set_xlabel("genotypes")

        return fig, ax

    def residuals(self, ax=None, stem=False, figsize=(6,4), axis=None, **kwargs):
        """ Get figure, return figure. """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        theory = self.model.statistics.predict()
        observed = self.model.phenotypes

        # Calculate residuals
        residuals =  theory - observed

        # Build array of residuals next to theoretical value
        # then sort those columns in ascending order of theoretical
        # value for plotting.
        data = np.array((theory, residuals))
        data = data[:, data[0, :].argsort()]

        ylim = max([abs(min(residuals)), abs(max(residuals))])

        # Create a stem plot of the data
        if stem:
            markerline, stemlines, baseline = ax.stem(data[0], data[1], markerfmt=" ", linewidth=6, color='b')
            plt.setp(markerline, 'markerfacecolor', 'b')
            plt.setp(stemlines, 'linewidth', 1.5)
            plt.setp(baseline, 'color','r', 'linewidth', 1)
        else:
            #for i in range(len(data[0])):
             #   print(str(data[0,i]) + "," + str(data[1,i]) + ",")
            ax.plot(data[0], data[1], **kwargs)
            ax.hlines(0, min(data[0]), max(data[0]), linestyle="dotted")
        ax.set_ylim([-ylim, ylim])

        if axis is not None:
            ax.axis(axis)

        return fig, ax

    def best_fit(self, ax=None, figsize=(6,4), errorbars=False, axis=None, **kwargs):
        """ Plot model line through date. """

        # Add to axis if given, else create new plot.
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Plot line through data
        theory = self.model.statistics.predict()
        observed = self.model.phenotypes

        # Sort the theory for line plot
        sorted_theory = np.sort(theory)

        # Plot the line through data

        if errorbars is True:

                if self.model.log_transform:
                    upper = np.log10(1 + self.model.stdeviations/self.model.Raw.phenotypes)
                    lower = np.log10(1 - self.model.stdeviations/self.model.Raw.phenotypes)

                else:
                    upper = self.stdeviations
                    lower = upper
                ax.errorbar(theory, observed, yerr=[upper,abs(lower)], **kwargs)
        else:
            ax.plot(theory, observed, **kwargs)
        ax.plot(sorted_theory, sorted_theory, color="r", linewidth=2)

        if axis is not None:
            ax.axis(axis)

        return fig, ax

    def summary(self, ):
        """ Plot a summary of the model. Includes a plot of
            the model with residuals plotted underneath.
        """
        pass
