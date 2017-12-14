# Denoise-py

Preprocessor to denoise measured field strength H, flux density B, magnetic polarization J = B - µ0 H, and magnetostriction λ for galfenol data.

This is needed especially for the magnetostriction, for which the raw data can be very noisy.

# Howto

With the MATLAB `.mat` data files in the current directory, run `python3 main.py` in the terminal.

Note that currently only the code is hosted here - you will need to obtain the data files separately.

Several files can be processed as a batch. For input file `foo.mat`, the output is `foo_denoised.mat`.

The individual filters, such as `filter_magnetostriction.py`, also have their own command-line interfaces. When run individually, they will display the filter result, along with a plot of the power spectral density that can be used to judge the quality of both the input and the output.

`singlevaluize.py` is meant as the next preprocessing step, to generate a reasonable single-valued BH curve (collapsing the hysteresis loop), but at the moment this does not work properly for the data it is meant for. Currently this part is done in MATLAB.

# Converting the data

Load the original raw data into Origin Viewer. The first four columns must be (H, B, J, λ), in that order.

Then: Select all, Edit ⊳ Copy full precision, paste into a text editor. Search-and-replace all `,` by `.`, to have a decimal separator that Python natively understands. Save.

This produces a suitably formatted text file that can be converted to be read by the filters. `convert_txt_to_mat.py` is a small standalone utility that performs the conversion.

For interoperability with MATLAB, the file format used by the filters is a MATLAB `.mat` file, containing the data in a 2D array called `A`, with the columns in the same order as in the original data.

The filters only use the first four columns; any excess columns are discarded by the filters.

## Text file format

`convert_txt_to_mat.py` expects a text file containing numbers. Columns must be separated by tabs (exactly one real tab, `U+0009`; not just any whitespace), and rows by newlines. Example:
```
17.0    -1.0    42.0    23.0
3.14    2.718    299792458.0    7.777
```
where the whitespace on each line represents a single tab character between each pair of columns.

The row index is simply the index of the sample in the discrete time signal.

In the output, any missing data (indicated by two or more consecutive tab characters) will be replaced by `NaN`.

# Dependencies

 - [NumPy](http://www.numpy.org)
 - [SciPy](https://scipy.org)
 - [scikit-image](http://scikit-image.org/) (used for total-variation denoising for 1D time signals)
 - [Matplotlib](http://matplotlib.org) (for viewing the result)
 
# License

[BSD](LICENSE.md)

