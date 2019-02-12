Development code for statistics of Level-2 SST CDR data and uncertainties.

Upon installation with pip, all the Python-based dependencies should be
installed automatically.  In addition, you will need:

- CDR L2P data in CF1.6 netCDF format, obtainable from ESA CCI SST.
- a L4 OSTIA netCDF for extraction of the landsea_mask.

Before the first run you may have to update the firstline db::

    import sst_cdr_stats

After installation, command-line utilities include:

    plot_sst
    plot_landsea_mask

Most of those have an online help, i.e. plot_sst --help, listing all the options and capabilities.

To get started with SST CDR statistical countings, you need to set the paths correctly to the input and output directories. 

Contact Michael Taylor <michael.taylor@reading.ac.uk>.

