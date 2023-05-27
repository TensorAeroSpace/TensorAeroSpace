/*
 * File: x15_model_private.h
 *
 * Code generated for Simulink model 'x15_model'.
 *
 * Model version                  : 1.29
 * Simulink Coder version         : 8.9 (R2015b) 13-Aug-2015
 * C/C++ source code generated on : Sun May 21 08:07:14 2023
 *
 * Target selection: ert_shrlib.tlc
 * Embedded hardware selection: Specified
 * Emulation hardware selection:
 *    Differs from embedded hardware (MATLAB Host)
 * Code generation objectives: Unspecified
 * Validation result: Not run
 */

#ifndef RTW_HEADER_x15_model_private_h_
#define RTW_HEADER_x15_model_private_h_
#include "rtwtypes.h"

/* Private macros used by the generated code to access rtModel */
#ifndef rtmIsMajorTimeStep
# define rtmIsMajorTimeStep(rtm)       (((rtm)->Timing.simTimeStep) == MAJOR_TIME_STEP)
#endif

#ifndef rtmIsMinorTimeStep
# define rtmIsMinorTimeStep(rtm)       (((rtm)->Timing.simTimeStep) == MINOR_TIME_STEP)
#endif

#ifndef rtmGetTPtr
# define rtmGetTPtr(rtm)               ((rtm)->Timing.t)
#endif

#ifndef rtmSetTPtr
# define rtmSetTPtr(rtm, val)          ((rtm)->Timing.t = (val))
#endif

/* private model entry point functions */
extern void x15_model_derivatives(void);

#endif                                 /* RTW_HEADER_x15_model_private_h_ */

/*
 * File trailer for generated code.
 *
 * [EOF]
 */
