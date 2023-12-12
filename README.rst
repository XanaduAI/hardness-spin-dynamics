The hardness of quantum spin dynamics
=====================================


This repository contains code to reproduce the anticoncentration results in the paper Park, Casares, Arrazola, and Huh: "The hardness of quantum spin dynamics."
In the paper, we considered the XX and the Ising models. There are several relevant codes for each Hamiltonian.

Build and Compile
-----------------

The repository contains C++ and Python code, with proper Python binding for some C++ functions. We recommend using a Python virtual environment to run our code. After cloning the repository, one can use the following commands to install the Python binding.

.. code-block:: bash

   $ python3 -m venv env        # create virtual env
   $ source ./env/bin/activate  # activate virtual env
   $ pip install .              # install Python binding


For building C++ code, the following commands should work. We note that `Eigen <https://eigen.tuxfamily.org/index.php?title=Main_Page>`_ and `OpenBLAS <https://www.openblas.net/>`_ are required to compile to code. A compiler with proper C++20 support (e.g., GCC >= 10) is also required to compile the code correctly.

.. code-block:: bash

   $ mkdir Build && cd Build
   $ cmake ..
   $ make

See also `Kokkos compile guide <https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/Compiling.html>`_ for supported backends.


The XX model
------------

For $n < 10$, where $n$ is the number of spins in each part of the bipartite graph, we fully diagonalized the Hamiltonian to obtain the time evolution of the output probability, $p(x;J;t)$. For this purpose, two relevant source files, ``src/xx_time_evol_exact.cpp`` and ``python_src/xx_time_evol_exact.py`` are used.
The C++ code ``src/xx_time_evol_exact.cpp`` diagonalizes the constructed Hamiltonian using Eigen, whereas Python code ``python_src/xx_time_evol_exact.py`` utilizes JAX. Thus, our Python code is GPU enabled, which is significantly faster for larger $n$.


For $n = 10$, we implemented the time evolution using the second-order Trotter decomposition in ``src/xx_time_evol_trotter.cpp``. Our code is based on `PennyLane-Lightning-Kokkos <https://github.com/PennyLaneAI/pennylane-lightning>`_. We ran our code using NVidia A100, compiled with the CUDA backend of Kokkos.


For computing the output probability at times multiples of log of $n$, ``src/xx_at_log_time_trotter.cpp`` is used.


The Ising model
---------------

