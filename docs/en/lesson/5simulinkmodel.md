# Building the Simulink Model

## Creating the control system in Simulink

![Control system in Simulink](img/image017.png){ width=400 }

To create the control system in Simulink, follow these steps:

1. Add components from the Simulink library to the canvas:
   - Simulink/Continuous/State-Space
   - Simulink/Sources/Digital Clock
   - Simulink/Commonly Used Blocks/In1
   - Simulink/Commonly Used Blocks/Out1
2. Rename the In1/Out1 blocks to meaningful signal names.
3. Specify the State-Space block parameters (conveniently via MATLAB scripts).

![State-Space block](img/image018.png){ width=400 }
