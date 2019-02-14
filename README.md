<img alt="ESA CCI: SST CDR2.1_release stats" align="right" src="http://cci.esa.int/sites/default/files/esa_cci_sst_logo_0.GIF">

[![Build Status](https://travis-ci.org/patternizer/SST_CDR_STATS.svg?branch=master)](https://travis-ci.org/patternizer/SST_CDR_STATS)
[![Build status](https://ci.appveyor.com/api/projects/status/SST_CDR_STATS/branch/master?svg=true)](https://ci.appveyor.com/project/patternizer/SST_CDR_STATS-core)
[![codecov.io](https://codecov.io/github/patternizer/SST_CDR_STATS/coverage.svg?branch=master)](https://codecov.io/github/patternizer/SST_CDR_STATS?branch=master)
[![Documentation Status](https://readthedocs.org/projects/SST_CDR_STATS/badge/?version=latest)](http://SST_CDR_STATS.readthedocs.io/en/latest/?badge=latest)

# SST_CDR_STATS

Development code for calculation of statistics of Level-2 SST CDR2.1_release data and uncertainties.

## Getting started

To get started with SST_CDR_STATS, the paths to the Level-2 daily orbit files from AVHRR and ATSR need setting as well as the path to the output directory. Running join_sst.sh will generate the daily netCDF summaries. Runtime failures can be fixed by running fix_nc.sh. 

After installation, command-line utilities include:

    plot_sst.py

which has the option to include a plot of the landsea_mask and ocean fraction:

    plot_landsea_mask.py

## Development

### Contributors

Thanks go to the members of the [ESA SST CCI project consortium](http://esa-sst-cci.org/?q=project%20team) for making the data required available. 

### Unit-testing

Unit testing will be performed using `pytest` and its coverage plugin `pytest-cov`.

To run the unit-tests with coverage, type

    $ export NUMBA_DISABLE_JIT=1
    $ py.test --cov=SST_CDR_STATS test
    
We need to set environment variable `NUMBA_DISABLE_JIT` to disable JIT compilation by `numba`, so that 
coverage reaches the actual Python code. We use Numba's JIT compilation to speed up numeric Python 
number crunching code.

### Generating the Documentation

The documentation will be generated with the [Sphinx](http://www.sphinx-doc.org/en/stable/rest.html) tool to create
a [ReadTheDocs](http://SST_CDR_STATS.readthedocs.io/en/latest/?badge=latest). 
If there is a need to build the docs locally, some 
additional software packages are required:

    $ conda install sphinx sphinx_rtd_theme mock
    $ conda install -c conda-forge sphinx-argparse
    $ pip install sphinx_autodoc_annotation

To regenerate the HTML docs, type    
    
    $ cd doc
    $ make html

## License

The code is distributed under terms and conditions of the [MIT license](https://opensource.org/licenses/MIT).

## Contact information

* Michael Taylor (michael.taylor@reading.ac.uk)

