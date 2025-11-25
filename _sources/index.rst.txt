#######
PyKappa
#######

PyKappa is a Python package that provides a simulator of and interpreted interface to rule-based models of the variety specified by the `Kappa language <https://kappalanguage.org/>`_.
Supported are
   * programatic construction of :class:`systems <pykappa.system.System>` from Kappa strings,
   * on-the-fly manipulation of systems such as :class:`adding <pykappa.mixture.Mixture.add>` and :class:`removing <pykappa.mixture.Mixture.remove>` agents and :class:`editing <pykappa.system.System.add_rule>` rules,
   * :class:`monitoring <pykappa.system.Monitor>` of the history of observables,
   * cached :class:`tracking <pykappa.mixture.Mixture.apply_update>` of rule embeddings for algorithmically efficient simulation :class:`updates <pykappa.system.System.update>`,
   * optional :class:`transfer <pykappa.system.System.update_via_kasim>` to `KaSim <https://github.com/Kappa-Dev/KappaTools>`_ for faster compiled execution.
Visit the :doc:`examples <examples/index>` gallery to see how PyKappa can be used to simulate systems of molecular interactions such as :doc:`polymerization <examples/linear_polymerization>` and :doc:`gene regulation <examples/lac_operon>`.
See the `language manual <https://kappalanguage.org/sites/kappalanguage.org/files/inline-files/Kappa_Manual.pdf>`_ for a detailed description of the Kappa language.

PyKappa is available via pip:

.. code-block:: bash

   pip install pykappa


.. toctree::
   :maxdepth: 2
   :hidden:

   tutorial
   examples/index
   api/index
