import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


class multiplot:
    def __init__(self, num_rows, num_cols, **kwargs):
        if (num_rows % 1) != 0 or (num_cols % 1) != 0:
            raise ValueError("require integer row and column numbers")
        else:
            num_rows = int(num_rows)
            num_cols = int(num_cols)
        # mpl.rcParams['xtick.labelsize'] = 12
        # mpl.rcParams['ytick.labelsize'] = 12
        # mpl.rcParams['axes.labelsize'] = 12

        if "num_harmonics" not in kwargs:
            kwargs["num_harmonics"] = 4
        if "orientation" not in kwargs:
            kwargs["orientation"] = "portrait"
        if "row_spacing" not in kwargs:
            kwargs["row_spacing"] = 1
        if "font_size" not in kwargs:
            mpl.rcParams["font.size"] = 10
        else:
            mpl.rcParams["font.size"] = kwargs["font_size"]
        if "col_spacing" not in kwargs:
            kwargs["col_spacing"] = 1
        if "plot_width" not in kwargs:
            kwargs["plot_width"] = 2
        if "plot_height" not in kwargs:
            kwargs["plot_height"] = 1
        if "fourier_position" in kwargs:
            kwargs["fourier_plot"] = True
        if "fourier_plot" not in kwargs:
            kwargs["fourier_plot"] = False
        if "fourier_position" not in kwargs and kwargs["fourier_plot"] != False:
            kwargs["fourier_position"] = [1]

        if "distribution_position" in kwargs and kwargs["orientation"] == "portrait":
            raise NotImplementedError(
                "Haven't put portrait mode in yet for the distribution"
            )
        if "distribution_position" not in kwargs:
            kwargs["distribution_position"] = [-1]
        if (
            "distribution_position" in kwargs
            and type(kwargs["distribution_position"]) is not list
        ):
            kwargs["distribution_position"] = [kwargs["distribution_position"]]
        if (
            "fourier_position" in kwargs
            and type(kwargs["fourier_position"]) is not list
        ):
            kwargs["fourier_position"] = [kwargs["fourier_position"]]
        if "harmonic_position" not in kwargs:
            kwargs["harmonic_position"] = [-1]
        if type(kwargs["harmonic_position"]) is not list:
            kwargs["harmonic_position"] = [kwargs["harmonic_position"]]
        for pos in kwargs["harmonic_position"]:
            if kwargs["orientation"] == "landscape":
                if pos >= num_rows:
                    raise ValueError(
                        str(pos)
                        + " is greater than largest row number "
                        + str(num_rows)
                    )
            elif kwargs["orientation"] == "portrait":
                if pos >= num_cols:
                    raise ValueError(
                        str(pos)
                        + " is greater than largest column number "
                        + str(num_cols)
                    )
        y_dim = (kwargs["num_harmonics"] * num_rows * kwargs["plot_height"]) + (
            kwargs["row_spacing"] * (num_rows - 1)
        )
        x_dim = (kwargs["plot_width"] * num_cols) + (
            kwargs["col_spacing"] * (num_cols - 1)
        )
        gridspecification = mpl.gridspec.GridSpec(y_dim, x_dim)
        if kwargs["orientation"] == "landscape":
            total_axes = []
            axes = ["row" + str(x) for x in range(1, num_rows + 1)]

            for i in range(0, num_rows):
                row_axes = []
                if i in kwargs["harmonic_position"]:
                    for q in range(0, kwargs["num_harmonics"] * num_cols):
                        rowspan = kwargs["plot_height"]
                        colspan = kwargs["plot_width"]
                        loc = (
                            i
                            * (
                                (kwargs["plot_height"] * kwargs["num_harmonics"])
                                + kwargs["row_spacing"]
                            )
                            + (kwargs["plot_height"] * (q % kwargs["num_harmonics"])),
                            int(np.floor(q / kwargs["num_harmonics"]))
                            * (kwargs["plot_width"] + kwargs["col_spacing"]),
                        )
                        """
                        x position is calculated as the row (i)*total row width (number of harmonics + spacing) and then the position within the row
                        which will be (e.g. 1, 2, 3, 1,2, 3 for N=3)
                        y position is the column position (will go e.g. 0, 0, 0, 1, 1, 1 when N=3) * the column length (so col at col 0 it will be 0)
                        then for col 1, the y position will be the size of col 0
                        """

                        subplotspec = gridspecification.new_subplotspec(
                            loc, rowspan, colspan
                        )
                        ax = plt.subplot(subplotspec)
                        row_axes.append(ax)
                elif i in kwargs["distribution_position"]:
                    for q in range(0, 2 * num_cols):
                        if q % 2 == 0:
                            rowspan = kwargs["plot_height"]
                            colspan = kwargs["plot_width"]
                            loc = (
                                i * (kwargs["num_harmonics"] + kwargs["row_spacing"]),
                                int(np.floor(q / 2))
                                * (kwargs["plot_width"] + kwargs["col_spacing"]),
                            )
                            subplotspec = gridspecification.new_subplotspec(
                                loc, rowspan, colspan
                            )
                            ax = plt.subplot(subplotspec)
                            row_axes.append(ax)
                        else:
                            rowspan = kwargs["plot_height"] * (
                                kwargs["num_harmonics"] - 1
                            )
                            colspan = kwargs["plot_width"]
                            loc = (
                                i * (kwargs["num_harmonics"] + kwargs["row_spacing"])
                                + (q % 2),
                                int(np.floor(q / 2))
                                * (kwargs["plot_width"] + kwargs["col_spacing"]),
                            )
                            subplotspec = gridspecification.new_subplotspec(
                                loc, rowspan, colspan
                            )
                            ax = plt.subplot(subplotspec)
                            row_axes.append(ax)
                elif (
                    kwargs["fourier_plot"] != False and i in kwargs["fourier_position"]
                ):
                    if kwargs["num_harmonics"] % 2 == 1:
                        f_val = int(np.floor(kwargs["num_harmonics"] / 2) + 1)
                    else:
                        f_val = int(kwargs["num_harmonics"] / 2)
                    for q in range(0, 2 * num_cols):
                        loc = (
                            i * (kwargs["num_harmonics"] + kwargs["row_spacing"])
                            + ((q % 2) * (f_val)),
                            int(np.floor(q / 2))
                            * (kwargs["plot_width"] + kwargs["col_spacing"]),
                        )
                        rowspan = int(np.floor(kwargs["num_harmonics"] / 2))
                        colspan = kwargs["plot_width"]
                        subplotspec = gridspecification.new_subplotspec(
                            loc, rowspan, colspan
                        )
                        ax = plt.subplot(subplotspec)
                        row_axes.append(ax)
                else:
                    for j in range(0, num_cols):
                        loc = (
                            i
                            * (
                                (kwargs["plot_height"] * kwargs["num_harmonics"])
                                + kwargs["row_spacing"]
                            ),
                            j * (kwargs["plot_width"] + kwargs["col_spacing"]),
                        )
                        rowspan = kwargs["num_harmonics"] * kwargs["plot_height"]
                        colspan = kwargs["plot_width"]
                        subplotspec = gridspecification.new_subplotspec(
                            loc, rowspan, colspan
                        )
                        ax = plt.subplot(subplotspec)
                        row_axes.append(ax)
                total_axes.append(row_axes)
            self.total_axes = total_axes
            self.gridspec = gridspecification
            self.axes_dict = dict(zip(axes, total_axes))
        elif kwargs["orientation"] == "portrait":
            axes = ["col" + str(x) for x in range(1, num_cols + 1)]
            total_axes = []
            for i in range(0, num_cols):
                row_axes = []
                if i in kwargs["harmonic_position"]:
                    for j in range(0, num_rows):
                        for q in range(0, kwargs["num_harmonics"]):
                            loc = (
                                j
                                * (
                                    (kwargs["num_harmonics"] * kwargs["plot_height"])
                                    + kwargs["row_spacing"]
                                )
                                + (q * kwargs["plot_height"]),
                                i * (kwargs["plot_width"] + kwargs["col_spacing"]),
                            )
                            rowspan = kwargs["plot_height"]
                            colspan = kwargs["plot_width"]
                            subplotspec = gridspecification.new_subplotspec(
                                loc, rowspan, colspan
                            )
                            ax = plt.subplot(subplotspec)
                            row_axes.append(ax)
                elif (
                    kwargs["fourier_plot"] != False and i in kwargs["fourier_position"]
                ):
                    if kwargs["num_harmonics"] % 2 == 1:
                        f_val = int(np.floor(kwargs["num_harmonics"] / 2) + 1)
                    else:
                        f_val = int(kwargs["num_harmonics"] / 2)
                    for j in range(0, num_rows):
                        for q in range(0, 2):
                            loc = (
                                (
                                    j
                                    * (kwargs["num_harmonics"] + kwargs["row_spacing"])
                                    + (q * f_val)
                                ),
                                i * (kwargs["plot_width"] + kwargs["col_spacing"]),
                            )
                            rowspan = int(np.floor(kwargs["num_harmonics"] / 2))
                            colspan = kwargs["plot_width"]
                            subplotspec = gridspecification.new_subplotspec(
                                loc, rowspan, colspan
                            )
                            ax = plt.subplot(subplotspec)
                            row_axes.append(ax)
                else:
                    for j in range(0, num_rows):
                        loc = (
                            j
                            * (
                                (kwargs["num_harmonics"] * kwargs["plot_height"])
                                + kwargs["row_spacing"]
                            ),
                            i * (kwargs["plot_width"] + kwargs["col_spacing"]),
                        )
                        rowspan = kwargs["num_harmonics"] * kwargs["plot_height"]
                        colspan = kwargs["plot_width"]
                        subplotspec = gridspecification.new_subplotspec(
                            loc, rowspan, colspan
                        )
                        ax = plt.subplot(subplotspec)
                        row_axes.append(ax)
                total_axes.append(row_axes)
            self.gridspec = gridspecification
            self.axes_dict = dict(zip(axes, total_axes))
