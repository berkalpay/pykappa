PyKappa is a Python package for interpreted simulation and analysis of rule-based models of the variety specified by the `Kappa language <https://kappalanguage.org/>`_.
It supports
   * programatic construction of :class:`systems <pykappa.system.System>`;
   * manipulation of systems mid-simulation, such as by :class:`custom graph transformation <pykappa.system.System.apply>` and tuning parameters via declared :class:`variables <pykappa.system.System>`;
   * :class:`monitoring <pykappa.analysis.Monitor>` the history of observables;
   * cached :class:`tracking <pykappa.mixture.Mixture>` of rule embeddings for algorithmically efficient simulation :class:`updates <pykappa.system.System.update>`; and
   * :class:`passing systems<pykappa.system.System.update_via_kasim>` to `KaSim <https://github.com/Kappa-Dev/KappaTools>`_ for faster compiled execution.
Visit the :doc:`examples <examples/index>` gallery to see how PyKappa can be used to simulate systems of molecular interactions such as :doc:`polymerization <examples/linear_polymerization>` and :doc:`gene regulation <examples/lac_operon>`.
See the `language manual <https://kappalanguage.org/static/manual.pdf>`_ for a detailed description of the Kappa language.

PyKappa is available via pip:

.. code-block:: bash

   pip install pykappa


.. toctree::
   :maxdepth: 2
   :hidden:

   tutorial
   examples/index
   api/index
