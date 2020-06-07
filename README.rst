Armchair Analysis
=================

*Python wrapper for armchairanalysis.com NFL data*

Quick start
-----------

Requirements: Python with pandas_.

Install the latest release with pip_::

   git clone git@github.com:morelandjs/armchair_analysis.git && cd armchair_analysis
   pip install .
   
.. _pip: https://pip.pypa.io
.. _pandas: https://pandas.pydata.org/

Example Usage::
   
   from armchair_analysis.game_data import spread_data
   
   print(spread_data.dataframe)
