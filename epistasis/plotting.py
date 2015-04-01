import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Useful plots for analyzing data 
# from regression data.
# -----------------------------

def learn_vs_known(learned, known, title="Known vs. Learned"):
    """ Create a plot showing the learned data vs. known data. """
    
    fig, ax = plt.subplots(1,1, dpi=300)
    
    ax.plot(known, learned, '.b')
    ax.hold(True)
    x = np.linspace(min(known), max(known), 1000)
    ax.plot(x,x, '-r', linewidth=1)
    ax.set_xlabel("Known", fontsize = 14)
    ax.set_ylabel("Learned", fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    return fig
    
def residual_plot(learned, known, title="Residual Plot"):
    """ Generate a residual plot. """
    fig, ax = plt.subplots(1,1, dpi=300)
    
    ax.stem(known, (learned-known), 'b-', markerfmt='.')
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("True")
    ax.set_ylabel("Residuals")
    
    return fig
    
