Integrating your own Simulink model
======================================

C++ code generation
------------------

To support controlled plants from Simulink, the Simulink add-on Embedded Coder is required.

To convert a Simulink model to C code:

#. Use In1/Out1 blocks to describe the input and output parameters.

#. 	In the Simulink settings, select: Code Generation/System target file ert_shrlib.tlc.

	.. image:: img/cpp_gen.png
  		:width: 400
  		:alt: State-Space block

#. To build the model, use the keyboard shortcut ctrl+B. Alternatively, you can do this from the navigation panel by selecting "Build model". As a result, a folder with C++ code will appear in the directory where the model was located.

Integrating a Simulink model into Python
---------------------------------------

#. Create an .so file

   Integration of a Simulink model into Python is done using a DLL (dynamic-link library). To generate it, a gcc compiler is required.

   Enter the command

   .. code-block::

      gcc -shared -o model.so -fPIC *.c

   where .c refers to all files with the .c extension.

   An .so file will appear in the folder.

#. Describe the interaction interface

  The interaction interface is described for input and output parameters using ctypes.Structure and the type converter rtwtypes (tensoraerospace/aerospacemodel/model/rtwtypes.py).

  ```python

    class ExtY(ctypes.Structure):

    _fields_ = [
        ("name1", type_from_rtwtypes),
        ("name2", type_from_rtwtypes),
    ]

    The name and type can be found in the generated C file. The file should be called MODEL_NAME.h. In this file, find the description of External inputs, External outputs.

  The DLL file contains 3 functions:
    * MODEL_NAME_initialize - used to initialize the model
    * MODEL_NAME_step - used to compute the next model step;
      the step size equals dt, defined in the Simulink model parameters
    * MODEL_NAME_terminate - used to release model resources

Example of using a Simulink model with Python:

The model is located at https://github.com/tensoraerospace/simulink-example

	.. image:: img/model.png
  		:width: 400
  		:alt: Model

.. container:: cell code

   ```python

      import os
      import ctypes

      import matplotlib.pyplot as plt

      from rtwtypes import *

.. container:: cell code

   ```python

      class ExtY(ctypes.Structure):
          """
              Output parameters Simulink model
              (name, type)
          """
          _fields_ = [
              ("Wz", real_T),
              ("theta_big", real_T),
              ("H", real_T),
              ("alpha", real_T),
              ("theta_small", real_T),
          ]


      class ExtU(ctypes.Structure):
          """
              INput parameters Simulink model
              (name, type)
          """
          _fields_ = [
              ("ref_signal", real_T),
          ]

.. container:: cell code

   ```python

      dll_path = os.path.abspath("model.so")
      dll = ctypes.cdll.LoadLibrary(dll_path)

.. container:: cell code

   ```python

      X = ExtU.in_dll(dll, 'model_U')
      Y = ExtY.in_dll(dll, 'model_Y')

.. container:: cell code

   ```python

      model_initialize = dll.model_initialize
      model_step = dll.model_step
      model_terminate = dll.model_terminate

.. container:: cell code

   ```python

      model_initialize()

      wz = []
      theta_big = []
      H = []
      alpha = []
      theta_small = []

      for step in range(int(2100)):
          X.ref_signal = -0.1
          model_step()

          wz.append(Y.Wz)
          theta_big.append(Y.theta_big)
          H.append(Y.H)
          alpha.append(Y.alpha)
          theta_small.append(Y.theta_small)

      model_terminate()

   .. container:: output execute_result

      ::

         0

.. container:: cell code

   ```python

      plt.plot(wz)

      plt.ylabel('$w_z$, [rad/s]')

   .. container:: output execute_result

      ::

         Text(0, 0.5, '$w_z$, [rad/s]')

   .. container:: output display_data

      .. image:: img/wz.png

.. container:: cell code

   ```python

      plt.plot(H)

      plt.ylabel('H, [m]')

   .. container:: output execute_result

      ::

         Text(0, 0.5, 'H, [m]')

   .. container:: output display_data

      .. image:: img/h.png

.. container:: cell code

   ```python

      plt.plot(theta_big)

      plt.ylabel('$\Theta$, [rad]')

   .. container:: output execute_result

      ::

         Text(0, 0.5, '$\\Theta$, [rad]')

   .. container:: output display_data

      .. image:: img/theta_big.png

.. container:: cell code

   ```python

      plt.plot(theta_small)

      plt.ylabel(r'$\theta$, [rad]')

   .. container:: output execute_result

      ::

         Text(0, 0.5, '$\\theta$, [rad]')

   .. container:: output display_data

      .. image:: img/theta_small.png

.. container:: cell code

   ```python

      plt.plot(alpha)

      plt.ylabel(r'$\alpha$, [rad]')

   .. container:: output execute_result

      ::

         Text(0, 0.5, '$\\alpha$, [rad]')

   .. container:: output display_data

      .. image:: img/alpha.png
