/*
 * File: comsat_model.c
 *
 * Code generated for Simulink model 'comsat_model'.
 *
 * Model version                  : 1.29
 * Simulink Coder version         : 8.9 (R2015b) 13-Aug-2015
 * C/C++ source code generated on : Sun May 21 08:06:10 2023
 *
 * Target selection: ert_shrlib.tlc
 * Embedded hardware selection: Intel->x86-64 (Windows64)
 * Code generation objectives: Unspecified
 * Validation result: Not run
 */

#include "comsat_model.h"
#include "comsat_model_private.h"

/* Continuous states */
X_comsat_model_T comsat_model_X;

/* External inputs (root inport signals with auto storage) */
ExtU_comsat_model_T comsat_model_U;

/* External outputs (root outports fed by signals with auto storage) */
ExtY_comsat_model_T comsat_model_Y;

/* Real-time model */
RT_MODEL_comsat_model_T comsat_model_M_;
RT_MODEL_comsat_model_T *const comsat_model_M = &comsat_model_M_;

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
  int_T nXc = 3;
  rtsiSetSimTimeStep(si,MINOR_TIME_STEP);

  /* Save the state values at time t in y, we'll use x as ynew. */
  (void) memcpy(y, x,
                (uint_T)nXc*sizeof(real_T));

  /* Assumes that rtsiSetT and ModelOutputs are up-to-date */
  /* f0 = f(t,y) */
  rtsiSetdX(si, f0);
  comsat_model_derivatives();

  /* f(:,2) = feval(odefile, t + hA(1), y + f*hB(:,1), args(:)(*)); */
  hB[0] = h * rt_ODE3_B[0][0];
  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + (f0[i]*hB[0]);
  }

  rtsiSetT(si, t + h*rt_ODE3_A[0]);
  rtsiSetdX(si, f1);
  comsat_model_step();
  comsat_model_derivatives();

  /* f(:,3) = feval(odefile, t + hA(2), y + f*hB(:,2), args(:)(*)); */
  for (i = 0; i <= 1; i++) {
    hB[i] = h * rt_ODE3_B[1][i];
  }

  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + (f0[i]*hB[0] + f1[i]*hB[1]);
  }

  rtsiSetT(si, t + h*rt_ODE3_A[1]);
  rtsiSetdX(si, f2);
  comsat_model_step();
  comsat_model_derivatives();

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
void comsat_model_step(void)
{
  if (rtmIsMajorTimeStep(comsat_model_M)) {
    /* set solver stop time */
    if (!(comsat_model_M->Timing.clockTick0+1)) {
      rtsiSetSolverStopTime(&comsat_model_M->solverInfo,
                            ((comsat_model_M->Timing.clockTickH0 + 1) *
        comsat_model_M->Timing.stepSize0 * 4294967296.0));
    } else {
      rtsiSetSolverStopTime(&comsat_model_M->solverInfo,
                            ((comsat_model_M->Timing.clockTick0 + 1) *
        comsat_model_M->Timing.stepSize0 + comsat_model_M->Timing.clockTickH0 *
        comsat_model_M->Timing.stepSize0 * 4294967296.0));
    }
  }                                    /* end MajorTimeStep */

  /* Update absolute time of base rate at minor time step */
  if (rtmIsMinorTimeStep(comsat_model_M)) {
    comsat_model_M->Timing.t[0] = rtsiGetT(&comsat_model_M->solverInfo);
  }

  /* Outport: '<Root>/rho' incorporates:
   *  StateSpace: '<Root>/State-Space'
   */
  comsat_model_Y.rho = comsat_model_X.StateSpace_CSTATE[0];

  /* Outport: '<Root>/theta' incorporates:
   *  StateSpace: '<Root>/State-Space'
   */
  comsat_model_Y.theta = comsat_model_X.StateSpace_CSTATE[1];

  /* Outport: '<Root>/omega' incorporates:
   *  StateSpace: '<Root>/State-Space'
   */
  comsat_model_Y.omega = comsat_model_X.StateSpace_CSTATE[2];
  if (rtmIsMajorTimeStep(comsat_model_M)) {
    /* DigitalClock: '<Root>/Digital Clock' */
    comsat_model_Y.sim_time = (((comsat_model_M->Timing.clockTick1+
      comsat_model_M->Timing.clockTickH1* 4294967296.0)) * 0.1);
  }

  if (rtmIsMajorTimeStep(comsat_model_M)) {
    rt_ertODEUpdateContinuousStates(&comsat_model_M->solverInfo);

    /* Update absolute time for base rate */
    /* The "clockTick0" counts the number of times the code of this task has
     * been executed. The absolute time is the multiplication of "clockTick0"
     * and "Timing.stepSize0". Size of "clockTick0" ensures timer will not
     * overflow during the application lifespan selected.
     * Timer of this task consists of two 32 bit unsigned integers.
     * The two integers represent the low bits Timing.clockTick0 and the high bits
     * Timing.clockTickH0. When the low bit overflows to 0, the high bits increment.
     */
    if (!(++comsat_model_M->Timing.clockTick0)) {
      ++comsat_model_M->Timing.clockTickH0;
    }

    comsat_model_M->Timing.t[0] = rtsiGetSolverStopTime
      (&comsat_model_M->solverInfo);

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
      comsat_model_M->Timing.clockTick1++;
      if (!comsat_model_M->Timing.clockTick1) {
        comsat_model_M->Timing.clockTickH1++;
      }
    }
  }                                    /* end MajorTimeStep */
}

/* Derivatives for root system: '<Root>' */
void comsat_model_derivatives(void)
{
  XDot_comsat_model_T *_rtXdot;
  _rtXdot = ((XDot_comsat_model_T *) comsat_model_M->ModelData.derivs);

  /* Derivatives for StateSpace: '<Root>/State-Space' incorporates:
   *  Derivatives for Inport: '<Root>/ref_signal'
   */
  _rtXdot->StateSpace_CSTATE[0] = 0.0;
  _rtXdot->StateSpace_CSTATE[1] = 0.0;
  _rtXdot->StateSpace_CSTATE[2] = 0.0;
  _rtXdot->StateSpace_CSTATE[0] += comsat_model_X.StateSpace_CSTATE[1];
  _rtXdot->StateSpace_CSTATE[1] += 0.01036 * comsat_model_X.StateSpace_CSTATE[0];
  _rtXdot->StateSpace_CSTATE[1] += 0.7757 * comsat_model_X.StateSpace_CSTATE[2];
  _rtXdot->StateSpace_CSTATE[2] += -0.1775 * comsat_model_X.StateSpace_CSTATE[1];
  _rtXdot->StateSpace_CSTATE[2] += 0.1513 * comsat_model_U.ref_signal;
}

/* Model initialize function */
void comsat_model_initialize(void)
{
  /* Registration code */

  /* initialize real-time model */
  (void) memset((void *)comsat_model_M, 0,
                sizeof(RT_MODEL_comsat_model_T));

  {
    /* Setup solver object */
    rtsiSetSimTimeStepPtr(&comsat_model_M->solverInfo,
                          &comsat_model_M->Timing.simTimeStep);
    rtsiSetTPtr(&comsat_model_M->solverInfo, &rtmGetTPtr(comsat_model_M));
    rtsiSetStepSizePtr(&comsat_model_M->solverInfo,
                       &comsat_model_M->Timing.stepSize0);
    rtsiSetdXPtr(&comsat_model_M->solverInfo, &comsat_model_M->ModelData.derivs);
    rtsiSetContStatesPtr(&comsat_model_M->solverInfo, (real_T **)
                         &comsat_model_M->ModelData.contStates);
    rtsiSetNumContStatesPtr(&comsat_model_M->solverInfo,
      &comsat_model_M->Sizes.numContStates);
    rtsiSetNumPeriodicContStatesPtr(&comsat_model_M->solverInfo,
      &comsat_model_M->Sizes.numPeriodicContStates);
    rtsiSetPeriodicContStateIndicesPtr(&comsat_model_M->solverInfo,
      &comsat_model_M->ModelData.periodicContStateIndices);
    rtsiSetPeriodicContStateRangesPtr(&comsat_model_M->solverInfo,
      &comsat_model_M->ModelData.periodicContStateRanges);
    rtsiSetErrorStatusPtr(&comsat_model_M->solverInfo, (&rtmGetErrorStatus
      (comsat_model_M)));
    rtsiSetRTModelPtr(&comsat_model_M->solverInfo, comsat_model_M);
  }

  rtsiSetSimTimeStep(&comsat_model_M->solverInfo, MAJOR_TIME_STEP);
  comsat_model_M->ModelData.intgData.y = comsat_model_M->ModelData.odeY;
  comsat_model_M->ModelData.intgData.f[0] = comsat_model_M->ModelData.odeF[0];
  comsat_model_M->ModelData.intgData.f[1] = comsat_model_M->ModelData.odeF[1];
  comsat_model_M->ModelData.intgData.f[2] = comsat_model_M->ModelData.odeF[2];
  comsat_model_M->ModelData.contStates = ((X_comsat_model_T *) &comsat_model_X);
  rtsiSetSolverData(&comsat_model_M->solverInfo, (void *)
                    &comsat_model_M->ModelData.intgData);
  rtsiSetSolverName(&comsat_model_M->solverInfo,"ode3");
  rtmSetTPtr(comsat_model_M, &comsat_model_M->Timing.tArray[0]);
  comsat_model_M->Timing.stepSize0 = 0.1;

  /* states (continuous) */
  {
    (void) memset((void *)&comsat_model_X, 0,
                  sizeof(X_comsat_model_T));
  }

  /* external inputs */
  comsat_model_U.ref_signal = 0.0;

  /* external outputs */
  (void) memset((void *)&comsat_model_Y, 0,
                sizeof(ExtY_comsat_model_T));

  /* InitializeConditions for StateSpace: '<Root>/State-Space' */
  comsat_model_X.StateSpace_CSTATE[0] = 0.0;
  comsat_model_X.StateSpace_CSTATE[1] = 0.0;
  comsat_model_X.StateSpace_CSTATE[2] = 0.0587;
}

/* Model terminate function */
void comsat_model_terminate(void)
{
  /* (no terminate code required) */
}

/*
 * File trailer for generated code.
 *
 * [EOF]
 */
