from wordvector.utils import plot_with_matplotlib, plot_with_plotly

x_vals, y_vals, labels = reduce_dimensions(model)

try:
    get_ipython()
except Exception:
    plot_function = plot_with_matplotlib
else:
    plot_function = plot_with_plotly

plot_function(x_vals, y_vals, labels)