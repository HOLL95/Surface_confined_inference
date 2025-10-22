import matplotlib.pyplot as plt
from PIL import Image
import os
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


def save_and_chop(fig, filename, dpi=300, quality=100):
    """
    Save a matplotlib figure and reduce its file size by half.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The matplotlib figure to save
    filename : str
        Output filename (e.g., 'graph.png' or 'graph.jpg')
    dpi : int, optional
        Resolution for saving (default: 300)
    quality : int, optional
        JPEG quality 1-100 (default: 85, only for .jpg/.jpeg)

    """
    # Save the original figure
    temp_file = 'temp_' + filename
    fig.savefig(temp_file, dpi=dpi, bbox_inches='tight')
    
    # Get original size
    original_size = os.path.getsize(temp_file) / 1024  # in KB
    
    # Open and resize image to half dimensions
    img = Image.open(temp_file)
    new_width = img.width // 2
    new_height = img.height // 2
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Save with compression
    if filename.lower().endswith(('.jpg', '.jpeg')):
        img_resized.save(filename, 'JPEG', quality=quality, optimize=False)
    else:
        img_resized.save(filename, optimize=False)
    
    # Get reduced size
    reduced_size = os.path.getsize(filename) / 1024  # in KB
    
    # Clean up temp file
    os.remove(temp_file)
    
    