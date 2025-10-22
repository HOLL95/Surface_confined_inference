def add_panel_label(ax, label, x=-0.1, y=1.05, fontsize=14, fontweight='bold'):
    """
    Add a bolded letter label above a matplotlib axis.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to add the label to
    label : str
        The letter or text to display (e.g., 'A', 'B', 'C')
    x : float, optional
        Horizontal position in axis coordinates (default: -0.1)
    y : float, optional
        Vertical position in axis coordinates (default: 1.05)
    fontsize : int, optional
        Font size of the label (default: 14)
    fontweight : str, optional
        Font weight (default: 'bold')
    
    Returns:
    --------
    text : matplotlib.text.Text
        The text object created
    """
    text = ax.text(x, y, label, transform=ax.transAxes,
                   fontsize=fontsize, fontweight=fontweight,
                   va='bottom', ha='right')
    return text