import warnings

import matplotlib.pyplot as plt
import numpy as np


def nyquist(spectra, **kwargs):
    if "ax" not in kwargs:
        _,kwargs["ax"]=plt.subplots(1,1)
    if "scatter" not in kwargs:
        kwargs["scatter"]=0
    if "label" not in kwargs:
        kwargs["label"]=None
    if "linestyle" not in kwargs:
        kwargs["linestyle"]="-"
    if "marker" not in kwargs:
        kwargs["marker"]="o"
    if "colour" not in kwargs:
        kwargs["colour"]=None
    if "orthonormal" not in kwargs:
        kwargs["orthonormal"]=True
    if "lw" not in kwargs:
        kwargs["lw"]=1.5
    if "alpha" not in kwargs:
        kwargs["alpha"]=1
    if "line" not in kwargs:
        kwargs["line"]=True
    if "disable_negative" not in kwargs:
        kwargs["disable_negative"]=False
    elif kwargs["line"]==False:
        if kwargs["scatter"]==False:
            raise ValueError(r"Need one of 'line' or 'scatter' to not be False")
    if "markersize" not in kwargs:
        kwargs["markersize"]=20
    if "ylabel" not in kwargs or kwargs["ylabel"]==True:
        kwargs["ylabel"]="$-Z_{Im}$ ($\\Omega$)"
    if "xlabel" not in kwargs or kwargs["xlabel"]==True:
        kwargs["xlabel"]="$Z_{Re}$ ($\\Omega$)"
    ax=kwargs["ax"]
    imag_spectra_mean=np.mean(spectra[:,1])
    if kwargs["line"]==True:
        if imag_spectra_mean<0:

            ax.plot(spectra[:,0], -spectra[:,1], label=kwargs["label"], linestyle=kwargs["linestyle"], color=kwargs["colour"], lw=kwargs["lw"], alpha=kwargs["alpha"])
        else:
            if kwargs["disable_negative"]==True:
                ax.plot(spectra[:,0], -spectra[:,1], label=kwargs["label"], linestyle=kwargs["linestyle"], color=kwargs["colour"], lw=kwargs["lw"], alpha=kwargs["alpha"])
            else:
                warnings.warn("The imaginary portion of the data may be set to negative in your data!")
                ax.plot(spectra[:,0], spectra[:,1], label=kwargs["label"], linestyle=kwargs["linestyle"], color=kwargs["colour"], lw=kwargs["lw"], alpha=kwargs["alpha"])
    if kwargs["xlabel"]!=False:
        ax.set_xlabel(kwargs["xlabel"])
    if kwargs["ylabel"]!=False:
        ax.set_ylabel(kwargs["ylabel"])
    total_max=max(np.max(spectra[:,0]), np.max(-spectra[:,1]))
    if kwargs["orthonormal"]==True:
        ax.set_xlim([0, total_max+0.1*total_max])
        ax.set_ylim([0, total_max+0.1*total_max])
    if kwargs["scatter"]!=0:
        if imag_spectra_mean<0:
            ax.scatter(spectra[:,0][0::kwargs["scatter"]], -spectra[:,1][0::kwargs["scatter"]], marker=kwargs["marker"], color=kwargs["colour"], s=kwargs["markersize"])
        else:
            if kwargs["disable_negative"]==True:
                ax.scatter(spectra[:,0][0::kwargs["scatter"]], -spectra[:,1][0::kwargs["scatter"]], marker=kwargs["marker"], color=kwargs["colour"], s=kwargs["markersize"])
            else:
                ax.scatter(spectra[:,0][0::kwargs["scatter"]], spectra[:,1][0::kwargs["scatter"]], marker=kwargs["marker"], color=kwargs["colour"], s=kwargs["markersize"])
def bode(spectra,frequency, **kwargs):
    if "ax" not in kwargs:
        _,kwargs["ax"]=plt.subplots(1,1)
    if "label" not in kwargs:
        kwargs["label"]=None
    if "type" not in kwargs:
        kwargs["type"]="both"
    if "twinx" not in kwargs:
        kwargs["twinx"]=kwargs["ax"].twinx()
    if "data_type" not in kwargs:
        kwargs["data_type"]="complex"
    if "compact_labels" not in kwargs:
        kwargs["compact_labels"]=False
    if "lw" not in kwargs:
        kwargs["lw"]=1.5
    if "alpha" not in kwargs:
        kwargs["alpha"]=1
    if "scatter" not in kwargs:
        kwargs["scatter"]=False
    if "phase_correction" not in kwargs:
        kwargs["phase_correction"]=False
    if "no_labels" not in kwargs:
        kwargs["no_labels"]=False
    if "markersize" not in kwargs:
        kwargs["markersize"]=20
    if "line" not in kwargs:
        kwargs["line"]=True
    if "colour" not in kwargs:
        kwargs["colour"]=None
    elif kwargs["line"]==False:
        if kwargs["scatter"]==False:
            raise ValueError(r"Need one of 'line' or 'scatter' to not be False")
    if kwargs["data_type"]=="complex":
        if kwargs["phase_correction"] is not False:
            spectra[:,0]=np.subtract(spectra[:,0],kwargs["phase_correction"])
        spectra=[complex(x, y) for x,y in zip(spectra[:,0], spectra[:,1])]
        
        phase=np.angle(spectra, deg=True)#np.arctan(np.divide(-spectra[:,1], spectra[:,0]))*(180/math.pi)
        #print(np.divide(spectra[:,1], spectra[:,0]))
        magnitude=np.log10(np.abs(spectra))#np.add(np.square(spectra[:,0]), np.square(spectra[:,1]))
    elif kwargs["data_type"]=="phase_mag":
        phase=spectra[:,0]
        magnitude=spectra[:,1]
        if "data_is_log" not in kwargs:
            kwargs["data_is_log"]=True
        if kwargs["data_is_log"]==False:
            magnitude=np.log10(magnitude)
        
        
    ax=kwargs["ax"]
    ax.set_xlabel("$\\log_{10}$(Frequency)")
    x_freqs=np.log10(frequency)
    if kwargs["type"]=="both":
        twinx=kwargs["twinx"]
        if kwargs["no_labels"]!=True:
                if kwargs["compact_labels"]==False:
                    ax.set_ylabel("-Phase")
                    twinx.set_ylabel("Magnitude")
                else:
                    ax.text(x=-0.05, y=1.05, s="$-\\psi$", fontsize=12, transform=ax.transAxes)
                    ax.text(x=0.96, y=1.05, s="$\\log_{10}(|Z|) $", fontsize=12, transform=ax.transAxes)
        if kwargs["line"]==True:
            ax.plot(x_freqs, -phase, label=kwargs["label"], lw=kwargs["lw"], alpha=kwargs["alpha"], color=kwargs["colour"])
            
            twinx.plot(x_freqs, magnitude, linestyle="--", lw=kwargs["lw"], alpha=kwargs["alpha"], color=kwargs["colour"])
        if kwargs["scatter"] is not False:
            ax.scatter(x_freqs[0::kwargs["scatter"]], -phase[0::kwargs["scatter"]], s=kwargs["markersize"], color=kwargs["colour"],label=kwargs["label"], alpha=kwargs["alpha"])
            twinx.scatter(x_freqs[0::kwargs["scatter"]], magnitude[0::kwargs["scatter"]], marker="v", s=kwargs["markersize"], color=kwargs["colour"], alpha=kwargs["alpha"])
        
    elif kwargs["type"]=="phase":
        if kwargs["compact_labels"]==False:
            ax.set_ylabel("Phase")
        else:
                ax.text(x=-0.05, y=1.05, s="-$\\psi$", fontsize=12, transform=ax.transAxes)
        ax.plot(x_freqs, -phase, label=kwargs["label"], lw=kwargs["lw"], alpha=kwargs["alpha"], color=kwargs["colour"])

    elif kwargs["type"]=="magnitude":
        
        ax.plot(x_freqs, magnitude, label=kwargs["label"], lw=kwargs["lw"], alpha=kwargs["alpha"], color=kwargs["colour"])
        if kwargs["compact_labels"]==False:
            ax.set_ylabel("Magnitude")
        else:
                ax.text(x=-0.05, y=1.05, s="$|Z|$", fontsize=12, transform=ax.transAxes)
