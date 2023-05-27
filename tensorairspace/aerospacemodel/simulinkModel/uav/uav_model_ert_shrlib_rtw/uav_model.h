/*
 * File: uav_model.h
 *
 * Code generated for Simulink model 'uav_model'.
 *
 * Model version                  : 1.27
 * Simulink Coder version         : 8.9 (R2015b) 13-Aug-2015
 * C/C++ source code generated on : Sun May 21 08:05:18 2023
 *
 * Target selection: ert_shrlib.tlc
 * Embedded hardware selection: Intel->x86-64 (Windows64)
 * Code generation objectives: Unspecified
 * Validation result: Not run
 */

#ifndef RTW_HEADER_uav_model_h_
#define RTW_HEADER_uav_model_h_
#include <string.h>
#ifndef uav_model_COMMON_INCLUDES_
# define uav_model_COMMON_INCLUDES_
#include "rtwtypes.h"
#include "rtw_continuous.h"
#include "rtw_solver.h"
#endif                                 /* uav_model_COMMON_INCLUDES_ */

#include "uav_model_types.h"

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

/* Continuous states (auto storage) */
typedef struct {
  real_T u[4];                         /* '<Root>/State-Space' */
} X_uav_model_T;

/* State derivatives (auto storage) */
typedef struct {
  real_T u[4];                         /* '<Root>/State-Space' */
} XDot_uav_model_T;

/* State disabled  */
typedef struct {
  boolean_T u[4];                      /* '<Root>/State-Space' */
} XDis_uav_model_T;

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
  real_T ref_signal;                   /* '<Root>/ref_signal' */
} ExtU_uav_model_T;

/* External outputs (root outports fed by signals with auto storage) */
typedef struct {
  real_T u;                            /* '<Root>/u' */
  real_T w;                            /* '<Root>/w' */
  real_T q;                            /* '<Root>/q' */
  real_T theta;                        /* '<Root>/theta' */
  real_T sim_time;                     /* '<Root>/sim_time ' */
} ExtY_uav_model_T;

/* Real-time Model Data Structure */
struct tag_RTM_uav_model_T {
  const char_T *errorStatus;
  RTWSolverInfo solverInfo;

  /*
   * ModelData:
   * The following substructure contains information regarding
   * the data used in the model.
   */
  struct {
    X_uav_model_T *contStates;
    int_T *periodicContStateIndices;
    real_T *periodicContStateRanges;
    real_T *derivs;
    boolean_T *contStateDisabled;
    boolean_T zCCacheNeedsReset;
    boolean_T derivCacheNeedsReset;
    boolean_T blkStateChange;
    real_T odeY[4];
    real_T odeF[3][4];
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
    uint32_T clockTickH0;
    time_T stepSize0;
    uint32_T clockTick1;
    uint32_T clockTickH1;
    SimTimeStep simTimeStep;
    boolean_T stopRequestedFlag;
    time_T *t;
    time_T tArray[2];
  } Timing;
};

/* Continuous states (auto storage) */
extern X_uav_model_T uav_model_X;

/* External inputs (root inport signals with auto storage) */
extern ExtU_uav_model_T uav_model_U;

/* External outputs (root outports fed by signals with auto storage) */
extern ExtY_uav_model_T uav_model_Y;

/* Model entry point functions */
extern void uav_model_initialize(void);
extern void uav_model_step(void);
extern void uav_model_terminate(void);

/* Real-time Model object */
extern RT_MODEL_uav_model_T *const uav_model_M;

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
 * '<Root>' : 'uav_model'
 */
#endif                                 /* RTW_HEADER_uav_model_h_ */

/*
 * File trailer for generated code.
 *
 * [EOF]
 */
