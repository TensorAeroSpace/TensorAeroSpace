/*
 * File: f16_model.c
 *
 * Code generated for Simulink model 'f16_model'.
 *
 * Model version                  : 1.27
 * Simulink Coder version         : 8.9 (R2015b) 13-Aug-2015
 * C/C++ source code generated on : Sun Dec 11 09:37:03 2022
 *
 * Target selection: ert_shrlib.tlc
 * Embedded hardware selection: Intel->x86-64 (Windows64)
 * Code generation objectives: Unspecified
 * Validation result: Not run
 */

#include "f16_model.h"
#include "f16_model_private.h"

/* Continuous states */
X_f16_model_T f16_model_X;

/* External inputs (root inport signals with auto storage) */
ExtU_f16_model_T f16_model_U;

/* External outputs (root outports fed by signals with auto storage) */
ExtY_f16_model_T f16_model_Y;

/* Real-time model */
RT_MODEL_f16_model_T f16_model_M_;
RT_MODEL_f16_model_T *const f16_model_M = &f16_model_M_;

/*
 * This function updates continuous states using the ODE3 fixed-step
 * solver algorithm
 */
static void rt_ertODEUpdateContinuousStates(RTWSolverInfo *si )
{
  /* Solver Matrices */
  static const real_T rt_ODE3_A[3] = {
    1.0/2.0, 3.0/4.0, 1.0
  };

  static const real_T rt_ODE3_B[3][3] = {
    { 1.0/2.0, 0.0, 0.0 },

    { 0.0, 3.0/4.0, 0.0 },

    { 2.0/9.0, 1.0/3.0, 4.0/9.0 }
  };

  time_T t = rtsiGetT(si);
  time_T tnew = rtsiGetSolverStopTime(si);
  time_T h = rtsiGetStepSize(si);
  real_T *x = rtsiGetContStates(si);
  ODE3_IntgData *id = (ODE3_IntgData *)rtsiGetSolverData(si);
  real_T *y = id->y;
  real_T *f0 = id->f[0];
  real_T *f1 = id->f[1];
  real_T *f2 = id->f[2];
  real_T hB[3];
  int_T i;
  int_T nXc = 4;
  rtsiSetSimTimeStep(si,MINOR_TIME_STEP);

  /* Save the state values at time t in y, we'll use x as ynew. */
  (void) memcpy(y, x,
                (uint_T)nXc*sizeof(real_T));

  /* Assumes that rtsiSetT and ModelOutputs are up-to-date */
  /* f0 = f(t,y) */
  rtsiSetdX(si, f0);
  f16_model_derivatives();

  /* f(:,2) = feval(odefile, t + hA(1), y + f*hB(:,1), args(:)(*)); */
  hB[0] = h * rt_ODE3_B[0][0];
  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + (f0[i]*hB[0]);
  }

  rtsiSetT(si, t + h*rt_ODE3_A[0]);
  rtsiSetdX(si, f1);
  f16_model_step();
  f16_model_derivatives();

  /* f(:,3) = feval(odefile, t + hA(2), y + f*hB(:,2), args(:)(*)); */
  for (i = 0; i <= 1; i++) {
    hB[i] = h * rt_ODE3_B[1][i];
  }

  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + (f0[i]*hB[0] + f1[i]*hB[1]);
  }

  rtsiSetT(si, t + h*rt_ODE3_A[1]);
  rtsiSetdX(si, f2);
  f16_model_step();
  f16_model_derivatives();

  /* tnew = t + hA(3);
     ynew = y + f*hB(:,3); */
  for (i = 0; i <= 2; i++) {
    hB[i] = h * rt_ODE3_B[2][i];
  }

  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + (f0[i]*hB[0] + f1[i]*hB[1] + f2[i]*hB[2]);
  }

  rtsiSetT(si, tnew);
  rtsiSetSimTimeStep(si,MAJOR_TIME_STEP);
}

/* Model step function */
void f16_model_step(void)
{
  if (rtmIsMajorTimeStep(f16_model_M)) {
    /* set solver stop time */
    if (!(f16_model_M->Timing.clockTick0+1)) {
      rtsiSetSolverStopTime(&f16_model_M->solverInfo,
                            ((f16_model_M->Timing.clockTickH0 + 1) *
        f16_model_M->Timing.stepSize0 * 4294967296.0));
    } else {
      rtsiSetSolverStopTime(&f16_model_M->solverInfo,
                            ((f16_model_M->Timing.clockTick0 + 1) *
        f16_model_M->Timing.stepSize0 + f16_model_M->Timing.clockTickH0 *
        f16_model_M->Timing.stepSize0 * 4294967296.0));
    }
  }                                    /* end MajorTimeStep */

  /* Update absolute time of base rate at minor time step */
  if (rtmIsMinorTimeStep(f16_model_M)) {
    f16_model_M->Timing.t[0] = rtsiGetT(&f16_model_M->solverInfo);
  }

  /* Outport: '<Root>/u' incorporates:
   *  StateSpace: '<Root>/State-Space'
   */
  f16_model_Y.u = f16_model_X.u[0];

  /* Outport: '<Root>/alpha' incorporates:
   *  StateSpace: '<Root>/State-Space'
   */
  f16_model_Y.alpha = f16_model_X.u[1];

  /* Outport: '<Root>/q' incorporates:
   *  StateSpace: '<Root>/State-Space'
   */
  f16_model_Y.q = f16_model_X.u[2];

  /* Outport: '<Root>/theta' incorporates:
   *  StateSpace: '<Root>/State-Space'
   */
  f16_model_Y.theta = f16_model_X.u[3];
  if (rtmIsMajorTimeStep(f16_model_M)) {
    /* DigitalClock: '<Root>/Digital Clock' */
    f16_model_Y.sim_time = (((f16_model_M->Timing.clockTick1+
      f16_model_M->Timing.clockTickH1* 4294967296.0)) * 0.1);
  }

  if (rtmIsMajorTimeStep(f16_model_M)) {
    rt_ertODEUpdateContinuousStates(&f16_model_M->solverInfo);

    /* Update absolute time for base rate */
    /* The "clockTick0" counts the number of times the code of this task has
     * been executed. The absolute time is the multiplication of "clockTick0"
     * and "Timing.stepSize0". Size of "clockTick0" ensures timer will not
     * overflow during the application lifespan selected.
     * Timer of this task consists of two 32 bit unsigned integers.
     * The two integers represent the low bits Timing.clockTick0 and the high bits
     * Timing.clockTickH0. When the low bit overflows to 0, the high bits increment.
     */
    if (!(++f16_model_M->Timing.clockTick0)) {
      ++f16_model_M->Timing.clockTickH0;
    }

    f16_model_M->Timing.t[0] = rtsiGetSolverStopTime(&f16_model_M->solverInfo);

    {
      /* Update absolute timer for sample time: [0.1s, 0.0s] */
      /* The "clockTick1" counts the number of times the code of this task has
       * been executed. The resolution of this integer timer is 0.1, which is the step size
       * of the task. Size of "clockTick1" ensures timer will not overflow during the
       * application lifespan selected.
       * Timer of this task consists of two 32 bit unsigned integers.
       * The two integers represent the low bits Timing.clockTick1 and the high bits
       * Timing.clockTickH1. When the low bit overflows to 0, the high bits increment.
       */
      f16_model_M->Timing.clockTick1++;
      if (!f16_model_M->Timing.clockTick1) {
        f16_model_M->Timing.clockTickH1++;
      }
    }
  }                                    /* end MajorTimeStep */
}

/* Derivatives for root system: '<Root>' */
void f16_model_derivatives(void)
{
  XDot_f16_model_T *_rtXdot;
  _rtXdot = ((XDot_f16_model_T *) f16_model_M->ModelData.derivs);

  /* Derivatives for StateSpace: '<Root>/State-Space' incorporates:
   *  Derivatives for Inport: '<Root>/ref_signal'
   */
  _rtXdot->u[0] = 0.0;
  _rtXdot->u[1] = 0.0;
  _rtXdot->u[2] = 0.0;
  _rtXdot->u[3] = 0.0;
  _rtXdot->u[0] += -0.1656 * f16_model_X.u[0];
  _rtXdot->u[0] += -10.7137 * f16_model_X.u[1];
  _rtXdot->u[0] += -7.2815 * f16_model_X.u[2];
  _rtXdot->u[0] += -32.174 * f16_model_X.u[3];
  _rtXdot->u[1] += -0.0018 * f16_model_X.u[0];
  _rtXdot->u[1] += -0.0981 * f16_model_X.u[1];
  _rtXdot->u[1] += 0.9276 * f16_model_X.u[2];
  _rtXdot->u[2] += -0.6252 * f16_model_X.u[1];
  _rtXdot->u[2] += -0.4673 * f16_model_X.u[2];
  _rtXdot->u[3] += f16_model_X.u[2];
  _rtXdot->u[0] += -4.0478 * f16_model_U.ref_signal;
  _rtXdot->u[1] += -0.0253 * f16_model_U.ref_signal;
  _rtXdot->u[2] += -0.8992 * f16_model_U.ref_signal;
}

/* Model initialize function */
void f16_model_initialize(void)
{
  /* Registration code */

  /* initialize real-time model */
  (void) memset((void *)f16_model_M, 0,
                sizeof(RT_MODEL_f16_model_T));

  {
    /* Setup solver object */
    rtsiSetSimTimeStepPtr(&f16_model_M->solverInfo,
                          &f16_model_M->Timing.simTimeStep);
    rtsiSetTPtr(&f16_model_M->solverInfo, &rtmGetTPtr(f16_model_M));
    rtsiSetStepSizePtr(&f16_model_M->solverInfo, &f16_model_M->Timing.stepSize0);
    rtsiSetdXPtr(&f16_model_M->solverInfo, &f16_model_M->ModelData.derivs);
    rtsiSetContStatesPtr(&f16_model_M->solverInfo, (real_T **)
                         &f16_model_M->ModelData.contStates);
    rtsiSetNumContStatesPtr(&f16_model_M->solverInfo,
      &f16_model_M->Sizes.numContStates);
    rtsiSetNumPeriodicContStatesPtr(&f16_model_M->solverInfo,
      &f16_model_M->Sizes.numPeriodicContStates);
    rtsiSetPeriodicContStateIndicesPtr(&f16_model_M->solverInfo,
      &f16_model_M->ModelData.periodicContStateIndices);
    rtsiSetPeriodicContStateRangesPtr(&f16_model_M->solverInfo,
      &f16_model_M->ModelData.periodicContStateRanges);
    rtsiSetErrorStatusPtr(&f16_model_M->solverInfo, (&rtmGetErrorStatus
      (f16_model_M)));
    rtsiSetRTModelPtr(&f16_model_M->solverInfo, f16_model_M);
  }

  rtsiSetSimTimeStep(&f16_model_M->solverInfo, MAJOR_TIME_STEP);
  f16_model_M->ModelData.intgData.y = f16_model_M->ModelData.odeY;
  f16_model_M->ModelData.intgData.f[0] = f16_model_M->ModelData.odeF[0];
  f16_model_M->ModelData.intgData.f[1] = f16_model_M->ModelData.odeF[1];
  f16_model_M->ModelData.intgData.f[2] = f16_model_M->ModelData.odeF[2];
  f16_model_M->ModelData.contStates = ((X_f16_model_T *) &f16_model_X);
  rtsiSetSolverData(&f16_model_M->solverInfo, (void *)
                    &f16_model_M->ModelData.intgData);
  rtsiSetSolverName(&f16_model_M->solverInfo,"ode3");
  rtmSetTPtr(f16_model_M, &f16_model_M->Timing.tArray[0]);
  f16_model_M->Timing.stepSize0 = 0.1;

  /* states (continuous) */
  {
    (void) memset((void *)&f16_model_X, 0,
                  sizeof(X_f16_model_T));
  }

  /* external inputs */
  f16_model_U.ref_signal = 0.0;

  /* external outputs */
  (void) memset((void *)&f16_model_Y, 0,
                sizeof(ExtY_f16_model_T));

  /* InitializeConditions for StateSpace: '<Root>/State-Space' */
  f16_model_X.u[0] = 160.0;
  f16_model_X.u[1] = 0.628;
  f16_model_X.u[2] = 0.0;
  f16_model_X.u[3] = 0.0;
}

/* Model terminate function */
void f16_model_terminate(void)
{
  /* (no terminate code required) */
}

/*
 * File trailer for generated code.
 *
 * [EOF]
 */
