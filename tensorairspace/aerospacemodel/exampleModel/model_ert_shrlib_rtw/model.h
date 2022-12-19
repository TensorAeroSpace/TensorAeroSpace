/*
 * File: model.h
 *
 * Code generated for Simulink model 'model'.
 *
 * Model version                  : 1.41
 * Simulink Coder version         : 8.9 (R2015b) 13-Aug-2015
 * C/C++ source code generated on : Sun Dec 18 21:44:12 2022
 *
 * Target selection: ert_shrlib.tlc
 * Embedded hardware selection: Intel->x86-64 (Windows64)
 * Code generation objectives: Unspecified
 * Validation result: Not run
 */

#ifndef RTW_HEADER_model_h_
#define RTW_HEADER_model_h_
#include <string.h>
#ifndef model_COMMON_INCLUDES_
# define model_COMMON_INCLUDES_
#include "rtwtypes.h"
#include "rtw_continuous.h"
#include "rtw_solver.h"
#endif                                 /* model_COMMON_INCLUDES_ */

#include "model_types.h"

/* Macros for accessing real-time model data structure */
#ifndef rtmGetErrorStatus
# define rtmGetErrorStatus(rtm)        ((rtm)->errorStatus)
#endif

#ifndef rtmSetErrorStatus
# define rtmSetErrorStatus(rtm, val)   ((rtm)->errorStatus = (val))
#endif

#ifndef rtmGetStopRequested
# define rtmGetStopRequested(rtm)      ((rtm)->Timing.stopRequestedFlag)
#endif

#ifndef rtmSetStopRequested
# define rtmSetStopRequested(rtm, val) ((rtm)->Timing.stopRequestedFlag = (val))
#endif

#ifndef rtmGetStopRequestedPtr
# define rtmGetStopRequestedPtr(rtm)   (&((rtm)->Timing.stopRequestedFlag))
#endif

#ifndef rtmGetT
# define rtmGetT(rtm)                  (rtmGetTPtr((rtm))[0])
#endif

/* Block signals (auto storage) */
typedef struct {
  real_T Integrator1;                  /* '<Root>/Integrator1' */
  real_T Sum3;                         /* '<Root>/Sum3' */
  real_T Sum5;                         /* '<Root>/Sum5' */
  real_T Gain6;                        /* '<Root>/Gain6' */
  real_T Sum2;                         /* '<Root>/Sum2' */
  real_T Saturation;                   /* '<S1>/Saturation' */
  real_T Sum;                          /* '<S1>/Sum' */
  real_T Sum1;                         /* '<Root>/Sum1' */
} B_model_T;

/* Continuous states (auto storage) */
typedef struct {
  real_T Integrator1_CSTATE;           /* '<Root>/Integrator1' */
  real_T Integrator4_CSTATE;           /* '<Root>/Integrator4' */
  real_T Integrator3_CSTATE;           /* '<Root>/Integrator3' */
  real_T Integrator_CSTATE;            /* '<Root>/Integrator' */
  real_T Integrator2_CSTATE;           /* '<Root>/Integrator2' */
  real_T Integrator2_CSTATE_j;         /* '<S1>/Integrator2' */
  real_T TransferFcn1_CSTATE;          /* '<Root>/Transfer Fcn1' */
  real_T TransferFcn_CSTATE;           /* '<S1>/Transfer Fcn' */
} X_model_T;

/* State derivatives (auto storage) */
typedef struct {
  real_T Integrator1_CSTATE;           /* '<Root>/Integrator1' */
  real_T Integrator4_CSTATE;           /* '<Root>/Integrator4' */
  real_T Integrator3_CSTATE;           /* '<Root>/Integrator3' */
  real_T Integrator_CSTATE;            /* '<Root>/Integrator' */
  real_T Integrator2_CSTATE;           /* '<Root>/Integrator2' */
  real_T Integrator2_CSTATE_j;         /* '<S1>/Integrator2' */
  real_T TransferFcn1_CSTATE;          /* '<Root>/Transfer Fcn1' */
  real_T TransferFcn_CSTATE;           /* '<S1>/Transfer Fcn' */
} XDot_model_T;

/* State disabled  */
typedef struct {
  boolean_T Integrator1_CSTATE;        /* '<Root>/Integrator1' */
  boolean_T Integrator4_CSTATE;        /* '<Root>/Integrator4' */
  boolean_T Integrator3_CSTATE;        /* '<Root>/Integrator3' */
  boolean_T Integrator_CSTATE;         /* '<Root>/Integrator' */
  boolean_T Integrator2_CSTATE;        /* '<Root>/Integrator2' */
  boolean_T Integrator2_CSTATE_j;      /* '<S1>/Integrator2' */
  boolean_T TransferFcn1_CSTATE;       /* '<Root>/Transfer Fcn1' */
  boolean_T TransferFcn_CSTATE;        /* '<S1>/Transfer Fcn' */
} XDis_model_T;

#ifndef ODE3_INTG
#define ODE3_INTG

/* ODE3 Integration Data */
typedef struct {
  real_T *y;                           /* output */
  real_T *f[3];                        /* derivatives */
} ODE3_IntgData;

#endif

/* External inputs (root inport signals with auto storage) */
typedef struct {
  real_T refSignal;                    /* '<Root>/refSignal' */
} ExtU_model_T;

/* External outputs (root outports fed by signals with auto storage) */
typedef struct {
  real_T Wz;                           /* '<Root>/Wz' */
  real_T theta_big;                    /* '<Root>/theta_big' */
  real_T H;                            /* '<Root>/H' */
  real_T alpha;                        /* '<Root>/alpha' */
  real_T theta_small;                  /* '<Root>/theta_small ' */
} ExtY_model_T;

/* Real-time Model Data Structure */
struct tag_RTM_model_T {
  const char_T *errorStatus;
  RTWSolverInfo solverInfo;

  /*
   * ModelData:
   * The following substructure contains information regarding
   * the data used in the model.
   */
  struct {
    X_model_T *contStates;
    int_T *periodicContStateIndices;
    real_T *periodicContStateRanges;
    real_T *derivs;
    boolean_T *contStateDisabled;
    boolean_T zCCacheNeedsReset;
    boolean_T derivCacheNeedsReset;
    boolean_T blkStateChange;
    real_T odeY[8];
    real_T odeF[3][8];
    ODE3_IntgData intgData;
  } ModelData;

  /*
   * Sizes:
   * The following substructure contains sizes information
   * for many of the model attributes such as inputs, outputs,
   * dwork, sample times, etc.
   */
  struct {
    int_T numContStates;
    int_T numPeriodicContStates;
    int_T numSampTimes;
  } Sizes;

  /*
   * Timing:
   * The following substructure contains information regarding
   * the timing information for the model.
   */
  struct {
    uint32_T clockTick0;
    time_T stepSize0;
    uint32_T clockTick1;
    SimTimeStep simTimeStep;
    boolean_T stopRequestedFlag;
    time_T *t;
    time_T tArray[2];
  } Timing;
};

/* Block signals (auto storage) */
extern B_model_T model_B;

/* Continuous states (auto storage) */
extern X_model_T model_X;

/* External inputs (root inport signals with auto storage) */
extern ExtU_model_T model_U;

/* External outputs (root outports fed by signals with auto storage) */
extern ExtY_model_T model_Y;

/* Model entry point functions */
extern void model_initialize(void);
extern void model_step(void);
extern void model_terminate(void);

/* Real-time Model object */
extern RT_MODEL_model_T *const model_M;

/*-
 * These blocks were eliminated from the model due to optimizations:
 *
 * Block '<Root>/Scope8' : Unused code path elimination
 * Block '<S1>/Scope' : Unused code path elimination
 * Block '<S1>/Scope2' : Unused code path elimination
 */

/*-
 * The generated code includes comments that allow you to trace directly
 * back to the appropriate location in the model.  The basic format
 * is <system>/block_name, where system is the system number (uniquely
 * assigned by Simulink) and block_name is the name of the block.
 *
 * Use the MATLAB hilite_system command to trace the generated code back
 * to the model.  For example,
 *
 * hilite_system('<S3>')    - opens system 3
 * hilite_system('<S3>/Kp') - opens and selects block Kp which resides in S3
 *
 * Here is the system hierarchy for this model
 *
 * '<Root>' : 'model'
 * '<S1>'   : 'model/Subsystem'
 */
#endif                                 /* RTW_HEADER_model_h_ */

/*
 * File trailer for generated code.
 *
 * [EOF]
 */
