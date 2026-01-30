% Options for packages loaded elsewhere
\PassOptionsToPackage{unicode}{hyperref}
\PassOptionsToPackage{hyphens}{url}
%
\documentclass[
]{article}
\usepackage{amsmath,amssymb}
\usepackage{lmodern}
\usepackage{iftex}
\ifPDFTeX
  \usepackage[T1]{fontenc}
  \usepackage[utf8]{inputenc}
  \usepackage{textcomp} % provide euro and other symbols
\else % if luatex or xetex
  \usepackage{unicode-math}
  \defaultfontfeatures{Scale=MatchLowercase}
  \defaultfontfeatures[\rmfamily]{Ligatures=TeX,Scale=1}
\fi
% Use upquote if available, for straight quotes in verbatim environments
\IfFileExists{upquote.sty}{\usepackage{upquote}}{}
\IfFileExists{microtype.sty}{% use microtype if available
  \usepackage[]{microtype}
  \UseMicrotypeSet[protrusion]{basicmath} % disable protrusion for tt fonts
}{}
\makeatletter
\@ifundefined{KOMAClassName}{% if non-KOMA class
  \IfFileExists{parskip.sty}{%
    \usepackage{parskip}
  }{% else
    \setlength{\parindent}{0pt}
    \setlength{\parskip}{6pt plus 2pt minus 1pt}}
}{% if KOMA class
  \KOMAoptions{parskip=half}}
\makeatother
\usepackage{xcolor}
\usepackage{longtable,booktabs,array}
\usepackage{multirow}
\usepackage{calc} % for calculating minipage widths
% Correct order of tables after \paragraph or \subparagraph
\usepackage{etoolbox}
\makeatletter
\patchcmd\longtable{\par}{\if@noskipsec\mbox{}\fi\par}{}{}
\makeatother
% Allow footnotes in longtable head/foot
\IfFileExists{footnotehyper.sty}{\usepackage{footnotehyper}}{\usepackage{footnote}}
\makesavenoteenv{longtable}
\usepackage{graphicx}
\makeatletter
\def\maxwidth{\ifdim\Gin@nat@width>\linewidth\linewidth\else\Gin@nat@width\fi}
\def\maxheight{\ifdim\Gin@nat@height>\textheight\textheight\else\Gin@nat@height\fi}
\makeatother
% Scale images if necessary, so that they will not overflow the page
% margins by default, and it is still possible to overwrite the defaults
% using explicit options in \includegraphics[width, height, ...]{}
\setkeys{Gin}{width=\maxwidth,height=\maxheight,keepaspectratio}
% Set default figure placement to htbp
\makeatletter
\def\fps@figure{htbp}
\makeatother
\setlength{\emergencystretch}{3em} % prevent overfull lines
\providecommand{\tightlist}{%
  \setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\setcounter{secnumdepth}{-\maxdimen} % remove section numbering
\ifLuaTeX
  \usepackage{selnolig}  % disable illegal ligatures
\fi
\IfFileExists{bookmark.sty}{\usepackage{bookmark}}{\usepackage{hyperref}}
\IfFileExists{xurl.sty}{\usepackage{xurl}}{} % add URL line breaks if available
\urlstyle{same} % disable monospaced font for URLs
\hypersetup{
  hidelinks,
  pdfcreator={LaTeX via pandoc}}

\author{}
\date{}

\begin{document}

\begin{longtable}[]{@{}
  >{\raggedright\arraybackslash}p{(\columnwidth - 2\tabcolsep) * \real{0.5000}}
  >{\raggedright\arraybackslash}p{(\columnwidth - 2\tabcolsep) * \real{0.5000}}@{}}
\toprule()
\begin{minipage}[b]{\linewidth}\raggedright
\includegraphics[width=0.92361in,height=0.75in]{vertopal_328f8d44c10a4dcd8c6d79eba0b32a72/media/image1.png}
\end{minipage} & \begin{minipage}[b]{\linewidth}\raggedright
\begin{quote}
Missouri University of Science and Technology
\end{quote}
\end{minipage} \\
\midrule()
\endhead
\bottomrule()
\end{longtable}

\begin{quote}
01 Sep 1997

Adaptive Critic Designs

Danil V. Prokhorov
\end{quote}

Donald C. Wunsch\\
Missouri University of Science and Technology, dwunsch@mst.edu

\begin{quote}
Follow this and additional works at:

\includegraphics[width=0.1875in,height=0.1875in]{vertopal_328f8d44c10a4dcd8c6d79eba0b32a72/media/image2.png}
Part of the

Recommended Citation

D. V. Prokhorov and D. C. Wunsch, "Adaptive Critic Designs," IEEE
Transactions on Neural Networks, vol. 8, no. 5, pp. 997-1007, Institute
of Electrical and Electronics Engineers (IEEE), Sep 1997.

The definitive version is available at

This Article - Journal is brought to you for free and open access by
Scholars\textquotesingle{} Mine. It has been accepted for inclusion in
Electrical and Computer Engineering Faculty Research \& Creative Works
by an authorized administrator of Scholars\textquotesingle{} Mine. This
work is protected by U. S. Copyright Law. Unauthorized use including
reproduction for redistribution requires the permission of the copyright
holder. For more information, please contact\\
.
\end{quote}

IEEE TRANSACTIONS ON NEURAL NETWORKS, VOL. 8, NO. 5, SEPTEMBER 1997 997

Adaptive Critic Designs

Danil V. Prokhorov, \emph{Student Member, IEEE,} and Donald C. Wunsch,
II, \emph{Senior Member, IEEE}

\textbf{\emph{Abstract---} We discuss a variety of adaptive critic
designs (ACD's) for neurocontrol. These are suitable for learning in
noisy, nonlinear, and nonstationary environments. They have common roots
as generalizations of dynamic programming for neural re-inforcement
learning approaches. Our discussion of these origins leads to an
explanation of three design families: Heuristic dy-namic programming
(HDP), dual heuristic programming (DHP), and globalized dual heuristic
programming (GDHP). The main emphasis is on DHP and GDHP as advanced
ACD's. We suggest two new modifications of the original GDHP design that
are currently the only working implementations of GDHP. They promise to
be useful for many engineering applications in the areas of optimization
and optimal control. Based on one of these modifications, we present a
unified approach to all ACD's. This leads to a generalized training
procedure for ACD's.}

\textbf{\emph{Index Terms---}Adaptive critic design (ACD),
backpropagation, control, DHP, dynamic programming, GDHP, HDP, heuristic
dynamic programming, neural network, neurocontrol, reinforce-ment
learning.}

\begin{quote}
I. ORIGINS OF ADAPTIVE CRITIC DESIGNS:

REINFORCEMENT LEARNING, DYNAMIC
\end{quote}

PROGRAMMING, AND BACKPROPAGATION\\
\textbf{R} also been a major focus for the neural-network community
physiologists since the time of Pavlov {[}1{]}, and has EINFORCEMENT
learning has been acknowledged by

{[}2{]}, {[}3{]}. At the time of these neural-network developments,

the existence of backpropagation {[}4{]}--{[}6{]}, was considered a

separate approach. Developments in the separate field of

dynamic programming {[}7{]}, {[}8{]}, led to a synthesis of all these

approaches. Early contributors to this synthesis included Wer-

bos {[}9{]}--{[}11{]}, Watkins {[}12{]}, {[}13{]}, and Barto \emph{et
al.} {[}14{]}. An even

earlier development by Widrow {[}15{]} explicitly implements a

critic neural element in a reinforcement learning problem.

\begin{quote}
To begin tracing these developments, consider the differ-
\end{quote}

ence between traditional supervised learning and traditional

reinforcement learning {[}16{]}. The former is a type of error-

based learning that was an outgrowth of simple perceptron

{[}17{]} or Adaline {[}18{]} networks. The latter is a form of match-

based learning that applies Hebbian learning {[}19{]}, and, in its

simplest manifestation, is a form of classical conditioning

{[}1{]}. Meanwhile, dynamic programming was attempting to

solve a problem that neither neural-network approach could

Manuscript received January 8, 1996; revised February 22, 1997. This
work was supported by the Texas Tech Center for Applied Research, Ford
Motor Co., and the National Science Foundation Neuroengineering Program
(Grant ECS-9413120).

The authors are with the Applied Computational Intelligence Laboratory,
Department of Electrical Engineering, Texas Tech University, Lubbock, TX
79409 USA.

\begin{quote}
Publisher Item Identifier S 1045-9227(97)05243-0.

handle. If we have a series of control actions that must be taken in
sequence, and we do not find out the quality of those actions until the
end of that sequence, how do we design an optimal controller? This is a
much harder problem than simply designing a controller to reach a set
point or maintain a reference trajectory. Although dynamic programming
can handle both deterministic and stochastic cases, here we illustrate
it in a deterministic context. Dynamic programming prescribes a search
tracking backward from the final step, rejecting all suboptimal paths
from any given point to the finish, but retaining all other possible
trajectories in memory until the starting point is reached. This can be
considered a ``smart'' exhaustive search in that all trajectories are
considered, but worthless ones are dropped at the earliest possible
point. However, many trajectories that are extremely unlikely to be
valuable are nonetheless retained until the search is complete. The
result of this is that the procedure is too computationally expensive
for most real problems. Moreover, the backward direction of the search
obviously precludes the use of dynamic programming in real-time control.

The other references cited above are to works that recog-nized the
fundamental idea of linking backpropagation with reinforcement learning
via a critic network. In supervised learning, a training algorithm
utilizes a desired output and, having compared it to the actual output,
generates an error term to allow the network to learn. It is convenient
to use back-propagation to get necessary derivatives of the error term
with respect to training parameters and/or inputs of the network. Here
we emphasize this interpretation of backpropagation merely as a tool of
getting required derivatives, rather than a complete training algorithm.

Critic methods remove the learning process one step from the control
network (traditionally called ``action network''or ``actor'' in ACD
literature), so that desired trajectory or control action information is
not necessary. The critic network learns to approximate the cost-to-go
or strategic utility func-tion (the function of the Bellman's equation
in dynamic programming) and uses the output of an action network as one
of its inputs, directly or indirectly. When the critic network learns,
backpropagation of error signals can continue along its input pathway
back to the action network. To the backpropagation algorithm, this input
pathway looks like just another synaptic connection that needs weight
adjustment. Thus, no desired action signal is needed. What is needed is
a desired cost function . However, because of various techniques for
stretching out a learning problem over time (e.g., {[}20{]} and
{[}21{]}), it is possible to use these methods without even knowing the
desired , but knowing the final cost and the
\end{quote}

1045--9227/97\$10.00~1997 IEEE

998 IEEE TRANSACTIONS ON NEURAL NETWORKS, VOL. 8, NO. 5, SEPTEMBER 1997

one-step cost (or its estimate) further referred to as the utility .
Thus, some of the architectures we will consider involve time-delay
elements.

The work of Barto \emph{et al.} {[}14{]} and that of Watkins {[}12{]}
both feature table-look up critic elements operating in discrete
domains. These do not have any backpropagation path to the action
network, but do use the action signals to estimate a utility or cost
function. Barto \emph{et al.} use an adaptive critic element for a
pole-balancing problem. Watkins {[}12{]} created the system known as
Q-learning (the name is taken from his notation), explicitly based on
dynamic programming. Wer-bos has championed a family of systems for
approximating dynamic programming {[}10{]}. His approach generalizes
previ-ously suggested designs for continuous domains. For example,
Q-learning becomes a special case of an action-dependent heuristic
dynamic programming (ADHDP; note the action-dependent prefix AD used
hereafter) in his family of systems. Werbos goes beyond a critic
approximating just the function . His systems called dual heuristic
programming (DHP) {[}23{]}, and globalized dual heuristic programming
(GDHP) {[}22{]} are developed to approximate derivatives of the function
with respect to the states, and both and its derivatives, respectively.
It should be pointed out that these systems do not require exclusively
neural-network implementations: any differentiable structure suffices as
a building block of the systems.

This paper focuses on DHP and GDHP and their AD forms as advanced ACD's,
although we start by describing simple ACD's: HDP and ADHDP (Section
II). We provide two new modifications of GDHP that are easier to
implement than the original GDHP design. We also introduce a new design
called ADGDHP, which is currently the topmost in the hierarchy of ACD's
(Section II-D). We show that our designs of GDHP and ADGDHP provide a
unified framework to all ACD's, i.e., any ACD can be obtained from them
by a simple reconfiguration. We propose a general training procedure for
adaptation of the networks of ACD in Section III. We contrast the
advanced ACD's with the simple ACD's in Section IV. In Section V, we
discuss results of experimental work.

II. DESIGN LADDER

\emph{A. HDP and ADHDP}

HDP and its AD form have a critic network that estimates the function
(cost-to-go) in the Bellman equation of dynamic programming, expressed
as follows:

(1)

where is a discount factor for finite horizon problems , and is the
utility function or local cost. The critic is trained forward in time,
which is of great importance for real-time operation. The critic network
tries to minimize the following error measure over time:

\begin{quote}
\includegraphics[width=2.91944in,height=1.54306in]{vertopal_328f8d44c10a4dcd8c6d79eba0b32a72/media/image3.png}

(a) (b)

Fig. 1. (a) Critic adaptation in ADHDP/HDP. This is the same critic
network in two consecutive moments in time. The critic's output J(t+1)
is necessary in order to give us the training signal\\
J(t + 1) + U(t), which is the target value for J(t). (b) Action
adaptation. R is a vector of observables, A is a control vector. We use
the constant @J=@J = 1 as the error signal in order to train the action
network to minimize J.

where stands for either a vector of observables of the plant (or the
states, if available) or a concatenation of and a control (or action)
vector . {[}The configuration for training the critic according to (3)
is shown in Fig. 1(a).{]} It should be noted that, although both\\
and depend on weights of the critic, we do not account for the
dependence of on weights while minimizing the error (2). For example, in
the case of minimization in the least mean squares (LMS) we could write
the following expression for the weights' update:
\end{quote}

(4)

\begin{quote}
where is a positive learning rate.1\\
We seek to minimize or maximize in the immediate future thereby
optimizing the overall cost expressed as a sum of all over the horizon
of the problem. To do so we need the action network connected as shown
in Fig. 1(b). To get with respect to the action's a gradient of the cost
function weights, we simply backpropagate (i.e., the constant 1) through
the network. This gives us and\\
for all inputs in the vector and all the action's weights ,
respectively.

In HDP, action-critic connections are mediated by a model (or
identification) network approximating dynamics of the plant. The model
is needed when the problem's temporal nature does not allow us to wait
for subsequent time steps to infer incremental costs. When we are able
to wait for this information or when sudden changes in plant dynamics
prevent us from using the same model, the action network is directly
connected to the critic network. This is called ADHDP.

\emph{B. DHP and ADDHP}

DHP and its AD form have a critic network that estimates the derivatives
of with respect to the vector . The critic
\end{quote}

\begin{longtable}[]{@{}
  >{\raggedright\arraybackslash}p{(\columnwidth - 2\tabcolsep) * \real{0.5000}}
  >{\raggedright\arraybackslash}p{(\columnwidth - 2\tabcolsep) * \real{0.5000}}@{}}
\toprule()
\begin{minipage}[b]{\linewidth}\raggedright
\begin{quote}
(2)\\
(3)
\end{quote}\strut
\end{minipage} & \begin{minipage}[b]{\linewidth}\raggedright
\begin{quote}
1There exists a formal argument on whether to disregard the dependence
of J{[}Y (t+1){]} on WC {[}24{]} or, on the contrary, to account for
such a dependence {[}25{]}. The former is our preferred way of adapting
WC throughout the paper since the latter seems to be more applicable for
finite-state Markov chains {[}8{]}.
\end{quote}
\end{minipage} \\
\midrule()
\endhead
\bottomrule()
\end{longtable}

PROKHOROV AND WUNSCH: ADAPTIVE CRITIC DESIGNS 999

network learns minimization of the following error measure over time:

(5)

where

(6)

\begin{longtable}[]{@{}
  >{\raggedright\arraybackslash}p{(\columnwidth - 2\tabcolsep) * \real{0.5000}}
  >{\raggedright\arraybackslash}p{(\columnwidth - 2\tabcolsep) * \real{0.5000}}@{}}
\toprule()
\begin{minipage}[b]{\linewidth}\raggedright
where
\end{minipage} & \begin{minipage}[b]{\linewidth}\raggedright
\begin{quote}
is a vector containing partial derivatives of
\end{quote}
\end{minipage} \\
\midrule()
\endhead
the scalar & \begin{minipage}[t]{\linewidth}\raggedright
\begin{quote}
with respect to the components of the vector
\end{quote}
\end{minipage} \\
\multicolumn{2}{@{}>{\raggedright\arraybackslash}p{(\columnwidth - 2\tabcolsep) * \real{1.0000} + 2\tabcolsep}@{}}{%
\begin{minipage}[t]{\linewidth}\raggedright
. The critic network's training is more complicated than in HDP since we
need to take into account all relevant pathways of backpropagation as
shown in Fig. 2, where the paths of derivatives and adaptation of the
critic are depicted by dashed lines.

\begin{quote}
In DHP, application of the chain rule for derivatives yields
\end{quote}
\end{minipage}} \\
\bottomrule()
\end{longtable}

(7)

\begin{longtable}[]{@{}
  >{\raggedright\arraybackslash}p{(\columnwidth - 2\tabcolsep) * \real{0.5000}}
  >{\raggedright\arraybackslash}p{(\columnwidth - 2\tabcolsep) * \real{0.5000}}@{}}
\toprule()
\multicolumn{2}{@{}>{\raggedright\arraybackslash}p{(\columnwidth - 2\tabcolsep) * \real{1.0000} + 2\tabcolsep}@{}}{%
\begin{minipage}[b]{\linewidth}\raggedright
where , and , are the

numbers of outputs of the model and the action networks,
\end{minipage}} \\
\midrule()
\endhead
respectively. By exploiting (7), each of vector from (6) is determined
by & components of the \\
\multicolumn{2}{@{}>{\raggedright\arraybackslash}p{(\columnwidth - 2\tabcolsep) * \real{1.0000} + 2\tabcolsep}@{}}{%
} \\
\bottomrule()
\end{longtable}

(8)

Action-dependent DHP (ADDHP) assumes direct connec-tion between the
action and the critic networks. However, unlike ADHDP, we still need to
have a model network because it is used for maintaining the pathways of
backpropagation. ADDHP can be readily obtained from our design of ADGDHP
to be discussed in the Section II-D.

The action network is adapted in Fig. 2 by propagating back through the
model down to the action. The\\
goal of such adaptation can be expressed as follows:

(9)

For instance, we could write the following expression for the weights'
update when applying the LMS training algorithm:

(10)

where is a positive learning rate.

\begin{quote}
\includegraphics[width=2.0875in,height=2.4in]{vertopal_328f8d44c10a4dcd8c6d79eba0b32a72/media/image4.png}

Fig. 2. Adaptation in DHP. This is the same critic network shown in two

consecutive moments in time. The discount factor is assumed to be equal

to 1. Pathways of backpropagation are shown by dashed lines. Components

of the vector(t + 1) are propagated back from outputs R(t + 1) of the

model network to its inputs R(t) and A(t), yielding the first term of
(7)

and the vector @J(t + 1)=@A(t), respectively. The latter is propagated
back

from outputs A(t) of the action network to its inputs R(t), completing
the

second term in (7). This corresponds to the left-hand backpropagation
pathway

(thicker line) in the figure. Backpropagation of the vector @U(t)=@A(t)

through the action network results in a vector with components computed

as the last term of (8). This corresponds to the right-hand
backpropagation

pathway from the action network (thinner line) in the figure. Following
(8),

the summator produces the error vector E2(t) used to adapt the critic
network.

The action network is adapted as follows. The vector(t + 1) is
propagated

back through the model network to the action network, and the resulting

vector is added to @U(t)=@A(t). Then an incremental adaptation of the
action

network is invoked with the goal (9).

\emph{C. GDHP}
\end{quote}

\begin{longtable}[]{@{}
  >{\raggedright\arraybackslash}p{(\columnwidth - 0\tabcolsep) * \real{1.0000}}@{}}
\toprule()
\begin{minipage}[b]{\linewidth}\raggedright
\begin{quote}
GDHP minimizes the error with respect to both and its derivatives. While
it is more complex to do this simultane-ously, the resulting behavior is
expected to be superior. We describe three ways to do GDHP (Figs. 3--5).
The first of these was proposed by Werbos in {[}22{]}. The other two are
our own new suggestions.

Training the critic network in GDHP utilizes an error measure which is a
combination of the error measures of HDP and DHP (2) and (5). This
results in the following LMS update rule for the critic's weights:
\end{quote}
\end{minipage} \\
\midrule()
\endhead
\bottomrule()
\end{longtable}

(11)

\begin{quote}
where is given in (8), and and are positive learning

rates.

A major source of additional complexity in GDHP

is the necessity of computing second-order derivatives
\end{quote}

. To get the adaptation signal-2 {[}the second

\begin{quote}
term in (11){]} in the originally proposed GDHP (Fig. 3), we first

need to create a network dual to our critic network. The dual

network inputs the output and states of all hidden neurons of

the critic. Its output, , is exactly what one would

get performing backpropagation from the critic's output to

its input . Here we need these computations performed

separately and explicitly shown as a dual network. Then we
\end{quote}

1000 IEEE TRANSACTIONS ON NEURAL NETWORKS, VOL. 8, NO. 5, SEPTEMBER 1997

\begin{quote}
\includegraphics[width=2.58333in,height=1.70694in]{vertopal_328f8d44c10a4dcd8c6d79eba0b32a72/media/image5.png}
\end{quote}

Fig. 3. Critic's adaptation in the general GDHP design. X is a state
vector of the network.1 (Adaptation Signal-1) +2 (Adaptation Signal-2)
is the total adaptation signal {[}see (11){]}. The discount factor is
assumed to be equal to one. According to (3), the summator at the upper
center outputs the HDP-style error. Based on (6), the summator to the
right produces the DHP-style error vector. The mixed second-order
derivatives @2J(t)=@R(t)@WC are obtained by finding derivatives of
outputs @J(t)=@R(t) of the critic's dual network with respect to the
weights WC of the critic network itself. (This is symbolized by the
dashed arrow that starts from the encircled 1.) The multiplier performs
a scalar product of the vector (6) with an appropriate column of the
array @2J(t)=@R(t)@WC, as illustrated by (16) in the Example.

can get the second derivatives sought by a straightforward but careful
backpropagation all the way down through the dual network into the
critic network. This is symbolized by the dashed line starting from the
encircled 1 in Fig. 3.

We have recently proposed and successfully tested a GDHP design with
critic's training based on deriving explicit for-mulas for finding (Fig.
4) {[}28{]}, and, to the best of our knowledge, it is the first
published successful implementation of GDHP {[}34{]}. While this design
is more specialized than the original one, its code is less complex,
which is an important issue since correct implementation of the design
of Fig. 3 is not a trivial task. We illustrate how to obtain for the
critic's training of this GDHP design in an example below.

Finally, we have also suggested and are currently working on the
simplest GDHP design with a critic network as shown in Fig. 5 {[}42{]}.
Here the burden of computing the second derivatives is reduced to the
minimum by exploiting a critic network with both scalar output of the
esti- . Thus, the second derivatives mate and vector output of are
conveniently obtained through backpropagation.

We do not perform training of the action network through internal
pathways of the critic network of Fig. 5 leading from its output to the
input because it would be equivalent to going back to HDP. We already
have high quality estimates of as the critic's outputs in the DHP
portion of this GDHP design and therefore use them instead.2Thus, the
action's training is carried out only by the critic's outputs, precisely
as in DHP. However, the output implicitly affects the action's training
through the weights' sharing in the critic. Of course, we do use the
critic's internal pathways from its output to the input to train the
action network in the designs of Figs. 3 and 4.

\begin{quote}
\includegraphics[width=2.53056in,height=1.72639in]{vertopal_328f8d44c10a4dcd8c6d79eba0b32a72/media/image6.png}

Fig. 4. Critic adaptation in our simplified GDHP design. Unlike GDHP in
Fig. 3, here we use explicit formulas to compute all necessary
derivatives @2J(t)=@R(t)@WC.
\end{quote}

\includegraphics[width=1.7125in,height=0.82361in]{vertopal_328f8d44c10a4dcd8c6d79eba0b32a72/media/image7.png}

\begin{longtable}[]{@{}
  >{\raggedright\arraybackslash}p{(\columnwidth - 4\tabcolsep) * \real{0.3333}}
  >{\raggedright\arraybackslash}p{(\columnwidth - 4\tabcolsep) * \real{0.3333}}
  >{\raggedright\arraybackslash}p{(\columnwidth - 4\tabcolsep) * \real{0.3333}}@{}}
\toprule()
\begin{minipage}[b]{\linewidth}\raggedright
Fig. 5.
\end{minipage} &
\multicolumn{2}{>{\raggedright\arraybackslash}p{(\columnwidth - 4\tabcolsep) * \real{0.6667} + 2\tabcolsep}@{}}{%
\begin{minipage}[b]{\linewidth}\raggedright
\begin{quote}
Critic network in a straightforward GDHP design.
\end{quote}
\end{minipage}} \\
\midrule()
\endhead
\multicolumn{3}{@{}>{\raggedright\arraybackslash}p{(\columnwidth - 4\tabcolsep) * \real{1.0000} + 4\tabcolsep}@{}}{%
\begin{minipage}[t]{\linewidth}\raggedright
\begin{quote}
\emph{Example:} This example illustrates how to calculate the mixed
second-order derivatives for the GDHP design of Fig. 4. We consider a
simple critic network shown in Fig. 6. It consists of two sigmoidal
neurons in its
\end{quote}
\end{minipage}} \\
\multicolumn{2}{@{}>{\raggedright\arraybackslash}p{(\columnwidth - 4\tabcolsep) * \real{0.6667} + 2\tabcolsep}}{%
only hidden layer and a linear output} &
\begin{minipage}[t]{\linewidth}\raggedright
\begin{quote}
. This network is
\end{quote}
\end{minipage} \\
\multicolumn{3}{@{}>{\raggedright\arraybackslash}p{(\columnwidth - 4\tabcolsep) * \real{1.0000} + 4\tabcolsep}@{}}{%
\begin{minipage}[t]{\linewidth}\raggedright
\begin{quote}
equivalent to the following function:
\end{quote}
\end{minipage}} \\
\bottomrule()
\end{longtable}

(12)

\begin{longtable}[]{@{}
  >{\raggedright\arraybackslash}p{(\columnwidth - 4\tabcolsep) * \real{0.3333}}
  >{\raggedright\arraybackslash}p{(\columnwidth - 4\tabcolsep) * \real{0.3333}}
  >{\raggedright\arraybackslash}p{(\columnwidth - 4\tabcolsep) * \real{0.3333}}@{}}
\toprule()
\begin{minipage}[b]{\linewidth}\raggedright
\begin{quote}
Derivatives
\end{quote}
\end{minipage} & \begin{minipage}[b]{\linewidth}\raggedright
,
\end{minipage} & \begin{minipage}[b]{\linewidth}\raggedright
\begin{quote}
, are obtained as follows:
\end{quote}
\end{minipage} \\
\midrule()
\endhead
\bottomrule()
\end{longtable}

(13)

\begin{longtable}[]{@{}
  >{\raggedright\arraybackslash}p{(\columnwidth - 2\tabcolsep) * \real{0.5000}}
  >{\raggedright\arraybackslash}p{(\columnwidth - 2\tabcolsep) * \real{0.5000}}@{}}
\toprule()
\begin{minipage}[b]{\linewidth}\raggedright
\begin{quote}
where
\end{quote}
\end{minipage} & \begin{minipage}[b]{\linewidth}\raggedright
\begin{quote}
is the Kronecker delta. We can get the mixed
\end{quote}
\end{minipage} \\
\midrule()
\endhead
\multicolumn{2}{@{}>{\raggedright\arraybackslash}p{(\columnwidth - 2\tabcolsep) * \real{1.0000} + 2\tabcolsep}@{}}{%
\begin{minipage}[t]{\linewidth}\raggedright
\begin{quote}
second-order derivatives with respect to the weights of the output
neuron as follows:
\end{quote}
\end{minipage}} \\
\bottomrule()
\end{longtable}

(14)

\begin{longtable}[]{@{}
  >{\raggedright\arraybackslash}p{(\columnwidth - 4\tabcolsep) * \real{0.3333}}
  >{\raggedright\arraybackslash}p{(\columnwidth - 4\tabcolsep) * \real{0.3333}}
  >{\raggedright\arraybackslash}p{(\columnwidth - 4\tabcolsep) * \real{0.3333}}@{}}
\toprule()
\begin{minipage}[b]{\linewidth}\raggedright
\begin{quote}
where
\end{quote}
\end{minipage} & \begin{minipage}[b]{\linewidth}\raggedright
\begin{quote}
, and
\end{quote}
\end{minipage} &
\multirow{2}{*}{\begin{minipage}[b]{\linewidth}\raggedright
. For the hidden layer
\end{minipage}} \\
\multicolumn{2}{@{}>{\raggedright\arraybackslash}p{(\columnwidth - 4\tabcolsep) * \real{0.6667} + 2\tabcolsep}}{%
\begin{minipage}[b]{\linewidth}\raggedright
neurons, the required derivatives are
\end{minipage}} \\
\midrule()
\endhead
\multicolumn{3}{@{}>{\raggedright\arraybackslash}p{(\columnwidth - 4\tabcolsep) * \real{1.0000} + 4\tabcolsep}@{}}{%
} \\
\bottomrule()
\end{longtable}

\begin{longtable}[]{@{}
  >{\raggedright\arraybackslash}p{(\columnwidth - 2\tabcolsep) * \real{0.5000}}
  >{\raggedright\arraybackslash}p{(\columnwidth - 2\tabcolsep) * \real{0.5000}}@{}}
\toprule()
\begin{minipage}[b]{\linewidth}\raggedright
2This situation is typical when ACD's are used for optimal control. In
other application domains where the estimates of @J=@R obtained from the
HDP portion of the design may be of a better quality than those of the
DHP portion, the use of these more accurate estimates is preferable
{[}40{]}.
\end{minipage} & \begin{minipage}[b]{\linewidth}\raggedright
(15)
\end{minipage} \\
\midrule()
\endhead
\bottomrule()
\end{longtable}

PROKHOROV AND WUNSCH: ADAPTIVE CRITIC DESIGNS 1001

\begin{quote}
\includegraphics[width=2.12917in,height=1.12917in]{vertopal_328f8d44c10a4dcd8c6d79eba0b32a72/media/image8.png}
\end{quote}

\begin{longtable}[]{@{}
  >{\raggedright\arraybackslash}p{(\columnwidth - 8\tabcolsep) * \real{0.2000}}
  >{\raggedright\arraybackslash}p{(\columnwidth - 8\tabcolsep) * \real{0.2000}}
  >{\raggedright\arraybackslash}p{(\columnwidth - 8\tabcolsep) * \real{0.2000}}
  >{\raggedright\arraybackslash}p{(\columnwidth - 8\tabcolsep) * \real{0.2000}}
  >{\raggedright\arraybackslash}p{(\columnwidth - 8\tabcolsep) * \real{0.2000}}@{}}
\toprule()
\multicolumn{5}{@{}>{\raggedright\arraybackslash}p{(\columnwidth - 8\tabcolsep) * \real{1.0000} + 8\tabcolsep}@{}}{%
\begin{minipage}[b]{\linewidth}\raggedright
Fig. 6. A simple network for the example of computing the second-order

derivatives @2J(t)=@R(t)@WC in our GDHP design given in Fig. 4.
\end{minipage}} \\
\midrule()
\endhead
where & , & , & , and & . Thus, \\
\multicolumn{5}{@{}>{\raggedright\arraybackslash}p{(\columnwidth - 8\tabcolsep) * \real{1.0000} + 8\tabcolsep}@{}}{%
based on (11), we can adapt weights in the network using the following
expression:} \\
\bottomrule()
\end{longtable}

(16)

where the indexes and are chosen appropriately. We also assume that
either\\
, or since is a constant bias term.

The example above can be easily generalized to larger networks.

It is clear that HDP and DHP can be readily obtained from a GDHP design
with the critic of Fig. 5. The simplicity and versatility of this GDHP
design is very appealing, and it prompted us to a straightforward
generalization of the critic of Fig. 5 for AD forms of ACD's. Thus, we
propose action-dependent GDHP (ADGDHP), to be discussed next.

\emph{D. ADGDHP}

As all AD forms of ACD's, ADGDHP features a direct connection between
the action and the critic networks. Fig. 7 shows adaptation processes in
ADGDHP. Although one could utilize critics similar with those
illustrated in Figs. 3 and 4, we found ADGDHP easier to demonstrate when
a critic akin to one of Fig. 5 is used. In addition, we gained
versatility in that the design of Fig. 7 can be readily transformed into
ADHDP or ADDHP.

\begin{quote}
Consider training of the critic network. We can write

\includegraphics[width=3.42917in,height=3.95694in]{vertopal_328f8d44c10a4dcd8c6d79eba0b32a72/media/image9.png}

Fig. 7. Adaptation in ADGDHP. The critic network outputs the scalar J
and two vectors,R andA. The vectorA(t + 1) backpropagated through the
action network is added toR(t + 1). The vectorR(t + 1) propagates back
through the model, then it is is split in two vectors. One of them goest
into the square summator to be added to the vector @U(t)=@R(t) and to
the rightmost term in (18) (not shown). The second vector is added to
the vector @U(t)=@R(t) in another summator. both of these summators
produce two appropriate error vectors E2(t), as in (19) and (20).
According to (3), the right oval summator computes the error E1(t). Two
error vectors E2(t) and the scalar E1(t) are used to train the critic
network. The action network is adap ted by the direct pathA(t + 1)
between the critic and the action networks.

where
\end{quote}

\begin{longtable}[]{@{}
  >{\raggedright\arraybackslash}p{(\columnwidth - 6\tabcolsep) * \real{0.2500}}
  >{\raggedright\arraybackslash}p{(\columnwidth - 6\tabcolsep) * \real{0.2500}}
  >{\raggedright\arraybackslash}p{(\columnwidth - 6\tabcolsep) * \real{0.2500}}
  >{\raggedright\arraybackslash}p{(\columnwidth - 6\tabcolsep) * \real{0.2500}}@{}}
\toprule()
\multicolumn{4}{@{}>{\raggedright\arraybackslash}p{(\columnwidth - 6\tabcolsep) * \real{1.0000} + 6\tabcolsep}@{}}{%
\begin{minipage}[b]{\linewidth}\raggedright
\end{minipage}} \\
\midrule()
\endhead
and & , &
\multicolumn{2}{>{\raggedright\arraybackslash}p{(\columnwidth - 6\tabcolsep) * \real{0.5000} + 2\tabcolsep}@{}}{%
\begin{minipage}[t]{\linewidth}\raggedright
\begin{quote}
are the numbers of outputs of the model and the
\end{quote}
\end{minipage}} \\
\multicolumn{4}{@{}>{\raggedright\arraybackslash}p{(\columnwidth - 6\tabcolsep) * \real{1.0000} + 6\tabcolsep}@{}}{%
\begin{minipage}[t]{\linewidth}\raggedright
\begin{quote}
action networks, respectively.

Based on (17) and (18), we obtain two error vectors,
\end{quote}
\end{minipage}} \\
\multicolumn{3}{@{}>{\raggedright\arraybackslash}p{(\columnwidth - 6\tabcolsep) * \real{0.7500} + 4\tabcolsep}}{%
and} & \begin{minipage}[t]{\linewidth}\raggedright
\begin{quote}
from (6) as follows:
\end{quote}
\end{minipage} \\
\bottomrule()
\end{longtable}

(19)

\begin{longtable}[]{@{}
  >{\raggedright\arraybackslash}p{(\columnwidth - 2\tabcolsep) * \real{0.5000}}
  >{\raggedright\arraybackslash}p{(\columnwidth - 2\tabcolsep) * \real{0.5000}}@{}}
\toprule()
\begin{minipage}[b]{\linewidth}\raggedright
(17)
\end{minipage} &
\multirow{2}{*}{\begin{minipage}[b]{\linewidth}\raggedright
\begin{quote}
(20) As in GDHP, the critic network is additionally trained by the
\end{quote}
\end{minipage}} \\
\multirow{2}{*}{\begin{minipage}[b]{\linewidth}\raggedright
(18)
\end{minipage}} \\
& \begin{minipage}[b]{\linewidth}\raggedright
\begin{quote}
scalar error according to (3). If one applies the LMS algorithm, it
results in an update rule similar to (11).

Fig. 7 also shows the direct adaptation path\\
between the action and the critic networks. We express the
\end{quote}\strut
\end{minipage} \\
\midrule()
\endhead
\bottomrule()
\end{longtable}

\begin{quote}
1002 IEEE TRANSACTIONS ON NEURAL NETWORKS, VOL. 8, NO. 5, SEPTEMBER 1997

goal of action's training as follows:
\end{quote}

(21)

\begin{quote}
Similar with what we stated in the Section II-C on GDHP,

training of the action network is not carried out through the

internal pathways of the critic network leading from its

output to the input since it would be equivalent to returning

to ADHDP. To train the action network, we use only the

critic's outputs so as to meet (21). The goal (21) is

the same for all AD forms of ACD's.

III. GENERAL TRAINING PROCEDURE AND RELATED ISSUES
\end{quote}

\begin{longtable}[]{@{}
  >{\raggedright\arraybackslash}p{(\columnwidth - 4\tabcolsep) * \real{0.3333}}
  >{\raggedright\arraybackslash}p{(\columnwidth - 4\tabcolsep) * \real{0.3333}}
  >{\raggedright\arraybackslash}p{(\columnwidth - 4\tabcolsep) * \real{0.3333}}@{}}
\toprule()
\multicolumn{3}{@{}>{\raggedright\arraybackslash}p{(\columnwidth - 4\tabcolsep) * \real{1.0000} + 4\tabcolsep}@{}}{%
\begin{minipage}[b]{\linewidth}\raggedright
\begin{quote}
This training procedure is a generalization of that suggested in
{[}26{]}, {[}30{]}, {[}33{]}, {[}38{]}, and {[}43{]}, and it is
applicable to any ACD. It consists of two training cycles: critic's and
action's. We always start with critic's adaptation alternating it with
action's until an acceptable performance is reached. We assume no
concurrent adaptation of the model network, which is previously trained
offline, and any reasonable initialization for\\
and . In the critic's training cycle, we carry out incremental
opti-mization of (2) and/or (5) by exploiting a suitable optimization
technique (e.g., LMS). We repeat the following operations times:
\end{quote}\strut
\end{minipage}} \\
\midrule()
\endhead
\multicolumn{2}{@{}>{\raggedright\arraybackslash}p{(\columnwidth - 4\tabcolsep) * \real{0.6667} + 2\tabcolsep}}{%
for HDP, DHP, GDHP} & for ADHDP, ADDHP, ADGDHP \\
1.0. & \begin{minipage}[t]{\linewidth}\raggedright
\begin{quote}
Initialize t = 0 and R(0)
\end{quote}
\end{minipage} & Initialize t = 0; R(0), and A(0) \\
1.1. & \begin{minipage}[t]{\linewidth}\raggedright
\begin{quote}
V (t) = fC{[}R(t); WC{]}
\end{quote}
\end{minipage} & \begin{minipage}[t]{\linewidth}\raggedright
\begin{quote}
V (t) = fC{[}R(t); A(t); WC{]}
\end{quote}
\end{minipage} \\
1.2. & \begin{minipage}[t]{\linewidth}\raggedright
\begin{quote}
A(t) = fA{[}R(t); WA{]}
\end{quote}
\end{minipage} & \begin{minipage}[t]{\linewidth}\raggedright
\begin{quote}
R(t + 1) =
\end{quote}
\end{minipage} \\
\multicolumn{3}{@{}>{\raggedright\arraybackslash}p{(\columnwidth - 4\tabcolsep) * \real{1.0000} + 4\tabcolsep}@{}}{%
fM{[}R(t); A(t); WM{]}} \\
1.3. & \begin{minipage}[t]{\linewidth}\raggedright
\begin{quote}
R(t + 1) =
\end{quote}
\end{minipage} & A(t + 1) = fA{[}R(t + 1); WA{]} \\
\multirow{2}{*}{1.4.} & \begin{minipage}[t]{\linewidth}\raggedright
\begin{quote}
fM{[}R(t); A(t); WM{]}
\end{quote}
\end{minipage} &
\multirow{2}{*}{\begin{minipage}[t]{\linewidth}\raggedright
\begin{quote}
V (t + 1) =
\end{quote}
\end{minipage}} \\
& V (t + 1) = fC{[}R(t + 1); WC{]} \\
1.5. &
\multicolumn{2}{>{\raggedright\arraybackslash}p{(\columnwidth - 4\tabcolsep) * \real{0.6667} + 2\tabcolsep}@{}}{%
\begin{minipage}[t]{\linewidth}\raggedright
\begin{quote}
fC{[}R(t + 1); A(t + 1); WC{]} Compute E1(t); E2(t) from (2) and/or (5),
and @V (t)=@WC, to be
\end{quote}
\end{minipage}} \\
\bottomrule()
\end{longtable}

\begin{quote}
used in an optimization algorithm, then invoke the algorithm to

perform one update of the critic's weights WC. For the update

example, see (4) and (11).

1.6. t = t + 1; continue from 1.1.

Here stands for or , , ,

and are the action, the critic and the model

networks, with their weights , respectively.

In the action's training cycle, we also carry out incremental

learning through an appropriate optimization routine, as in the

critic's training cycle above. The list of operations for the

action's training cycle is almost the same as that for the critic's

cycle above (lines 1.0--1.6). However, we need to use (9) or

(21), rather than (2) and/or (5); and instead of

before invoking the optimization algorithm for

updating the action's weights {[}see (10) for the update

example{]}.

The action's training cycle should be repeated times

while keeping the critic's weights fixed. We point out

that and are lengths of the corresponding training

cycles. They are problem-dependent parameters of loosely

specified values. If we can easily combine
\end{quote}

\includegraphics[width=3.55417in,height=1.97639in]{vertopal_328f8d44c10a4dcd8c6d79eba0b32a72/media/image10.png}

\begin{quote}
Fig. 8. Test results of the autolander problem given for one of the most
challenging cases where wind gusts were made 50\% stronger than in
standard conditions. After the ACD's were trained on the number of
landings shown, they were tested in 600 more trials, without any
adaptation. Although the average training is much longer for GDHP and
DHP than for HDP and ADHDP, we could not observe an improvement of
performance for either HDP or ADHDP if we continued their training
further. Tight success means landing within a shortened touchdown region
of the runway (it is the most important characteristic). Loose success
means landing within the limits of the standard runway. Similar results
were obtained in various other flight conditions.

both the cycles to avoid duplicating the computations in lines 1.1--1.4.
After the action's training cycle is completed, one may check action's
performance, then stop or continue the training procedure entering the
critic's training cycle again, if the performance is not acceptable
yet.3\\
It is very important that the whole system consisting of ACD and plant
would remain stable while both the networks of ACD undergo adaptation.
Regarding this aspect of the training procedure, we recommend to start
the first training cycle of the critic with the action network trained
beforehand to act as a stabilizing controller of the plant. Such a
pretraining could be done on a linearized model of the plant (see, e.g.,
{[}45{]}). Bradtke \emph{et al}. {[}26{]} proved that, in the case of
the well-known linear quadratic regulation, a linear critic network with
quadratic inputs trained by the recursive least squares algorithm in an
ADHDP design converges to the optimal cost. If the regulator always
outputs actions which are optimal with respect to the target vector for
the critic's adaptation, i.e.,
\end{quote}

(22)

\begin{quote}
where , then the sequence is stabilizing, and it converges to the
optimal control sequence.

Control sequences obtained through classical dynamic pro-gramming are
known to guarantee stable control, assuming a perfect match between the
actual plant and its model used in dynamic programming. Balakrishnan
\emph{et al}. {[}43{]} suggested to stretch this fact over to a
DHP-based ACD for linear and nonlinear control of systems with known
models. In their design, one performs a training procedure similar to
the

3Like many other training procedures, ours also implicitly assumes a
sufficiently varied set of training examples (e.g., different training
trajectories) repeated often enough in order to satisfy persistent
excitation---a property well known in a modern identification and
adaptive control literature (see, e.g., {[}37{]}).
\end{quote}

PROKHOROV AND WUNSCH: ADAPTIVE CRITIC DESIGNS 1003

above. Each training cycle is continued till convergence of the
network's weights (i.e., , in the procedure above). It is also suggested
to use a new randomly chosen on every return to the beginning of the
critic's training cycle (line 1.6 is modified as follows: ; continue
from 1.0). It is argued that whenever the action's weights converge one
has a stable control, and such a training procedure eventually finds the
optimal control sequence.

While theory behind classical dynamic programming de-mands choosing the
optimal vector of (22) for each training cycle of the action network, we
suggest incremental learning of the action network in the training
procedure above. A vector produced at the end of the action's training
cycle does not necessarily match the vector\\
. However, our experience {[}28{]}, {[}30{]}, {[}44{]}, {[}46{]}, along
with successful results in {[}33{]}, {[}38{]}, and {[}43{]}, indicates
that choosing\\
precisely is not critical.

No training procedure currently exists that explicitly ad-dresses issues
of an inaccurate or uncertain model . It appears that model network
errors of as much as 20\% are tolerable, and ACD's trained with such
inaccurate model networks are nevertheless sufficiently robust {[}30{]}.
Although it seems consistent with assessments of robustness of
con-ventional neurocontrol (model-reference control with neural
networks) {[}31{]}, {[}32{]}, further research on robustness of control
with ACD is needed, and we are currently pursuing this work. To allow
using the training procedure above in presence of the model network's
inaccuracies, we suggest to run the model network concurrently with the
actual plant or another model, which imitates the plant more accurately
than the model network but, unlike this network, it is not
differentiable. The plant's outputs are then fed into the model network
every so often (usually, every time step) to provide necessary
align-ments and prevent errors of multiple-step-ahead predictions from
accumulating. Such a concurrently running arrangement is known under
different names including teacher forcing {[}35{]} and series-parallel
model {[}36{]}. After this arrangement is incorporated in an ACD, the
critic will usually input the plant's outputs, rather than the predicted
ones from the model network. Thus, the model network is mainly utilized
and to calculate the auxiliary derivatives\\
.

\begin{quote}
IV. SIMPLE ACD'S VERSUS ADVANCED ACD'S
\end{quote}

The use of derivatives of an optimization criterion, rather than the
optimization criterion itself, is known as being the most important
information to have in order to find an ac-ceptable solution. In the
simple ACD's, HDP, and ADHDP, this information is obtained indirectly:
by backpropagation through the critic network. It has a potential
problem of being too coarse since the critic network in HDP is not
trained to approximate derivatives of\\
directly. An approach to improve accuracy of this approximation has been
proposed in {[}27{]}. It is suggested to explore a set of trajectories
bordering a volume around the nominal trajectory of the plant during the
critic's training, rather than the nominal trajectory alone. In spite of

\begin{quote}
this enhancement, we still expect better performance from the advanced
ACD's.

Furthermore, Baird {[}39{]} showed that the shorter the dis-cretization
interval becomes, the slower the training of AD-HDP proceeds. In
continuous time, it is completely incapable of learning.

DHP and ADDHP have an important advantage over the simple ACD's since
their critic networks build a representation for derivatives of by being
explicitly trained on them through and . For instance, in the area of
model-based control we usually have a sufficiently accurate model
network and well-defined and . To adapt the action network we ultimately
need the derivatives or , rather than the function itself. But an
approximation of these derivatives is already a \emph{direct} output of
the DHP and ADDHP critics. Although multilayer neural networks are well
known to be universal approximators of not only a function itself
(direct output of the network) but also its derivatives with respect to
the network's inputs (indirect output obtained through backpropagation)
{[}41{]}, we note that the quality of such a direct approximation is
always better than that of any indirect approximation for given sizes of
the network and the training data. Work on a formal proof of this
advantage of DHP and ADDHP is currently in progress, but the reader is
referred to the Section V for our experimental justification.

Critic networks in GDHP and ADGDHP directly approxi-mate not only the
function but also its derivatives. Knowing both and its derivatives is
useful in problems where avail-ability of global information associated
with the function , i.e., the itself is as important as knowledge of the
slope of\\
derivatives of {[}40{]}. Besides, any shift of attention paid to values
of or its derivatives during training can be readily accommodated by
selecting unequal learning rates and in (11) (see Section II-C). In
Section II-C we described three GDHP designs. While the design of Fig. 5
seems to be the most straightforward and beneficial from the viewpoint
of small computational expenses, the designs of Figs. 3 and 4 use the
critic network more efficiently.

Advanced ACD's include DHP, ADDHP, GDHP, and ADGDHP, the latter two
being capable of emulating all the previous ACD's. All these designs
assume availability of the model network. Along with direct
approximation of the derivatives of , it contributes to a superior
performance of advanced ACD's over simple ones (see the next Section for
examples of performance comparison). Although the final selection among
advanced ACD's should certainly be based on comparative results, we
believe that in many applications the use of DHP or ADDHP is quite
enough. We also note that the AD forms of the designs may have an
advantage in training recurrent action networks.
\end{quote}

V. EXPERIMENTAL STUDIES

\begin{quote}
This section provides an overview of our experimental work on applying
various ACD's to control of dynamic systems. For detailed information on
interesting experiments carried out by
\end{quote}

1004 IEEE TRANSACTIONS ON NEURAL NETWORKS, VOL. 8, NO. 5, SEPTEMBER 1997

\includegraphics[width=5.01667in,height=1.91667in]{vertopal_328f8d44c10a4dcd8c6d79eba0b32a72/media/image11.png}

(a)

\includegraphics[width=5.01667in,height=2.03611in]{vertopal_328f8d44c10a4dcd8c6d79eba0b32a72/media/image12.png}

(b)\\
Fig. 9. Test results of two neurocontrollers for the ball-and-beam
system. Edges of the beam correspond to1, and its center is at zero.
Dotted lines are the desired ball positions xd (set points), solid lines
are the actual ball trajectory x(t). (a) Conventional neurocontroller
trained by the truncated backpropagation through time with NDEKF; (b)
DHP action network tested on the same set points as in (a).

\begin{longtable}[]{@{}
  >{\raggedright\arraybackslash}p{(\columnwidth - 0\tabcolsep) * \real{1.0000}}@{}}
\toprule()
\begin{minipage}[b]{\linewidth}\raggedright
other researchers in the field, the reader is referred to {[}33{]} and
{[}43{]}.

The first problem deals with a simplified model of a com-mercial
aircraft which is to be landed in a specified touchdown region of a
runway within given ranges of speed and pitch angle {[}22{]}. The
aircraft is subject to wind disturbances that have two components: wind
shear (deterministic) and turbulent wind gusts (stochastic). To land
safely, an external controller should be developed to provide an
appropriate sequence of command elevator angles to the aircraft's pitch
autopilot. Along with actual states of the plane, a controller may also
use desired values of the altitude and the vertical speed supplied by an
instrument landing system (ILS).

To trade off between closely following the desired landing profile from
the ILS when far from the ground, and meeting the landing constraints at
the touchdown, one could use the following utility function:
\end{minipage} \\
\midrule()
\endhead
\bottomrule()
\end{longtable}

(23)

where , , are experimentally determined con-

stants, and , , and are the actual altitude,

vertical speed, and horizontal position of the plane. To avoid

\begin{quote}
a singularity at , we treat both terms as fixed to unity whenever ft.

We found the problem with its original system of con-straints not
challenging enough since even the nonadaptive PID controller provided in
{[}22{]} could solve it very well. We complicated the problem by
shortening the touchdown region of the runway by 30\%.

We have compared the PID controller, ADHDP, HDP, and DHP for the same
complicated version of the autolander problem. Implementation details
are discussed in {[}28{]} and {[}30{]}, and results are summarized in
Fig. 8. The most important conclusion is that in going from the simplest
ACD, ADHDP, to the more advanced ACD's one can attain a significant
improvement in performance.

We have also applied DHP to control of actual hardware, a ball-and-beam
system {[}44{]}.4The goal is to balance the ball at an arbitrary
specified location on the beam. We use the recurrent multilayer
perceptron for both model and action networks. The model network inputs
the current position of the ball, , and the servo motor control signal,
the latter being the only output of the action network with a sigmoidal
output node. It predicts the next ball position, . The action networks
inputs from the model network and , the desired ball position at the
next time step. The critic

4Although we initially attempted an HDP design, we failed to make it
work: its critic was not accurate enough to allow the action's training.
\end{quote}

PROKHOROV AND WUNSCH: ADAPTIVE CRITIC DESIGNS 1005

\includegraphics[width=6.29722in,height=1.32361in]{vertopal_328f8d44c10a4dcd8c6d79eba0b32a72/media/image13.png}

(a)

\includegraphics[width=6.31667in,height=1.20278in]{vertopal_328f8d44c10a4dcd8c6d79eba0b32a72/media/image14.png}

(b)

\includegraphics[width=6.29722in,height=1.18889in]{vertopal_328f8d44c10a4dcd8c6d79eba0b32a72/media/image15.png}

(c)

\includegraphics[width=6.31944in,height=1.13611in]{vertopal_328f8d44c10a4dcd8c6d79eba0b32a72/media/image16.png}

(d)\\
Fig. 10. Performance of HDP {[}plots (a) and (b){]} and DHP {[}(c) and
(d){]} for the MIMO plant. Dotted lines are the reference trajectories y
1and y 2, solid lines are the actual outputs y1(t) and y2(t). The rms
error for DHP is 0.32 versus 0.68 for HDP.

network uses and to produce an output, .

We trained the action network off-line using a sufficiently accurate
model network trained in the parallel identification scheme {[}36{]}. We
trained the DHP design according to the , training procedure described
in Section III. As the utility we have used the squared difference
between and . Training was performed using the node-decoupled extended
Kalman filter (NDEKF) algorithm {[}31{]}. The typical training
trajectory consisted of 300 consecutive points, with two or three
distinct desired locations of the ball. We were usually able to obtain
an acceptable controller after three alternating critic's and action's
training cycles. Starting with in (6), we moved on to\\
and 0.9 for the second and the third critic's cycles, respectively.

Fig. 9 shows a sample of performance of the DHP action network when
tested on the actual ball-and-beam system for three set points
\emph{not} used in training. For comparison,

\begin{quote}
performance of a conventional neurocontroller is also given. This
neurocontroller of the same architecture as the action network was
trained with the same model network by truncated backpropagation through
time with NDEKF {[}32{]}.

Another example experiment deals with a nonlinear
multiple-input/multiple-output (MIMO) system proposed by Narendra and
Mukhopadhyay {[}45{]} controlled by HDP and DHP designs {[}46{]}. This
plant has three states, two inputs, and two outputs, and it is highly
unstable for small input changes. The maximum time delay between the
first control input and the second output is equal to three time steps.
The goal is to develop a controller to track two independent reference
signals as closely as possible.

Although Narendra and Mukhopadhyay have explored sev-eral control cases,
here we discuss only the case of fully accessible states and known plant
equations. Thus, instead of the model network, we utilize plant
equations within the framework of both ACD's.
\end{quote}

1006 IEEE TRANSACTIONS ON NEURAL NETWORKS, VOL. 8, NO. 5, SEPTEMBER 1997

\begin{longtable}[]{@{}
  >{\raggedright\arraybackslash}p{(\columnwidth - 2\tabcolsep) * \real{0.5000}}
  >{\raggedright\arraybackslash}p{(\columnwidth - 2\tabcolsep) * \real{0.5000}}@{}}
\toprule()
\begin{minipage}[b]{\linewidth}\raggedright
\begin{quote}
The action network inputs the plant state variables, , and the desired
plant outputs
\end{quote}
\end{minipage} & \begin{minipage}[b]{\linewidth}\raggedright
\begin{quote}
, and
\end{quote}
\end{minipage} \\
\midrule()
\endhead
\multicolumn{2}{@{}>{\raggedright\arraybackslash}p{(\columnwidth - 2\tabcolsep) * \real{1.0000} + 2\tabcolsep}@{}}{%
\begin{minipage}[t]{\linewidth}\raggedright
, to be tracked by the actual plant outputs\\
and , respectively. Since we have different time delays for each control
input/plant output pair, we used the following utility:\strut
\end{minipage}} \\
\bottomrule()
\end{longtable}

(24)

The critic's input vector consists of , , , , , . Both the action and
the critic networks are simple feedforward multilayer perceptrons with
one hidden layer of only six nodes. This is a much smaller size than
that of the controller network used in {[}45{]}, and we attribute our
success in training to the NDEKF algorithm. The typical training
procedure lasted three alternations of critic's and action's training
cycles (see Section III). The action network was initially pretrained to
act as a stabilizing controller {[}45{]}, then the first critic's cycle
began with\\
in (6) on a 300-point trajectory.

Fig. 10 shows our results for both HDP and DHP. We continued training
both designs until their performance was no longer improving. The HDP
action network performed much worse than its DHP counterpart. Although
there is still room for improvement (e.g., using a larger network), we
doubt that HDP performance will ever be as good as that of DHP.
Recently, KrishnaKumar {[}47{]} has reported HDP performance better than
ours in Fig. 10(a) and (b). However, our DHP results in Fig. 10(c) and
(d) still remain superior. We think that this is a manifestation of an
intrinsically less accurate approximation of the derivatives of in HDP,
as stated in Section IV.

VI. CONCLUSION

We have discussed the origins of ACD's as a conjunction of
backpropagation, dynamic programming, and reinforcement learning. We
have shown ACD's through the design ladder with steps varying in both
complexity and power, from HDP to DHP, and to GDHP and its
action-dependent form at the highest level. We have unified and
generalized all ACD's via our interpretation of GDHP and ADGDHP.
Experiments with these ACD's have proven consistent with our assessment
of their relative capabilities.

ACKNOWLEDGMENT

The authors wish to thank Drs. P. Werbos and L. Feldkamp for stimulating
and helpful discussions.

REFERENCES

\begin{quote}
{[}1{]} I. P. Pavlov, \emph{Conditional Reflexes: An Investigation of
the Physiological Activity of the Cerebral Cortex.} London: Oxford Univ.
Press, 1927. {[}2{]} S. Grossberg, ``Pavlovian pattern learning by
nonlinear neural networks,''in \emph{Proc. Nat. Academy Sci.,} 1971, pp.
828--831.

{[}3{]} A. H. Klopf, \emph{The Hedonistic Neuron: A Theory of Memory,
Learning and Intelligence.} Washington, DC: Hemisphere, 1982.

{[}4{]} P. J. Werbos, ``Beyond regression: New tools for prediction and
analysis in the behavioral sciences,'' Ph.D. dissertation, Committee on
Appl. Math., Harvard Univ., Cambridge, MA, 1974.
\end{quote}

\begin{longtable}[]{@{}
  >{\raggedright\arraybackslash}p{(\columnwidth - 0\tabcolsep) * \real{1.0000}}@{}}
\toprule()
\begin{minipage}[b]{\linewidth}\raggedright
\begin{quote}
{[}5{]} , \emph{The Roots of Backpropagation: From Ordered Derivatives
to} \emph{Neural Networks and Political Forecasting.} New York: Wiley,
1994. {[}6{]} Y. Chauvin and D. Rumelhart, Eds., \emph{Backpropagation:
Theory, Archi-} \emph{tectures, and Applications.} Hillsdale, NJ:
Lawrence Erlbaum, 1995.

Princeton, NJ: Princeton Univ. {[}7{]} R. E. Bellman, \emph{Dynamic
Programming.}

Press, 1957.

{[}8{]} D. P. Bertsekas, \emph{Dynamic Programming: Deterministic and
Stochastic} \emph{Models.} Englewood Cliffs, NJ: Prentice-Hall, 1987.

{[}9{]} P. J. Werbos, ``The elements of intelligence,'' \emph{Cybern.,}
no. 3, 1968. {[}10{]} , ``Advanced forecasting methods for global crisis
warning and models of intelligence,'' \emph{General Syst. Yearbook,}
vol. 22, pp. 25--38, 1977.

{[}11{]} , ``Applications of advances in nonlinear sensitivity
analysis,'' in \emph{Proc. 10th IFIP Conf. Syst. Modeling and
Optimization,} R. F. Drenick and F. Kosin, Eds. NY: Springer-Verlag,
1982.

{[}12{]} C. Watkins, ``Learning from delayed rewards,'' Ph.D.
dissertation, Cam- bridge Univ., Cambridge, U.K., 1989.

{[}13{]} C. Watkins and P. Dayan, ``Q-learning,'' \emph{Machine
Learning,} vol. 8, pp. 279--292, 1992.

{[}14{]} A. G. Barto, R. S. Sutton, and C. W. Anderson, ``Neuronlike
elements that can solve difficult learning control problems,''
\emph{IEEE Trans. Syst., Man, Cybern.,} vol. SMC-13, pp. 835--846, 1983.

{[}15{]} B. Widrow, N. Gupta, and S. Maitra, ``Punish/reward: Learning
with a critic in adaptive threshold systems,'' \emph{IEEE Trans. Syst.,
Man, Cybern.,} vol. SMC-3, pp. 455--465, 1973.

{[}16{]} R. S. Sutton, \emph{Reinforcement Learning.} Boston, MA:
Kluwer, 1996. {[}17{]} F. Rosenblatt, \emph{Principles of
Neurodynamics.} Washington, D.C.: Spar- tan, 1962.

{[}18{]} B. Widrow and M. Lehr, ``30 years of adaptive neural networks:
Perceptron, madaline, and backpropagation,'' \emph{Proc. IEEE,} vol. 78,
no. 9, pp. 1415--1442, 1990.

{[}19{]} D. O. Hebb, \emph{The Organization of Behavior.} New York:
Wiley, 1949. {[}20{]} R. S. Sutton, ``Learning to predict by the methods
of temporal differ- ences,'' \emph{Machine Learning,} vol. 3, pp. 9--44,
1988.

{[}21{]} P. J. Werbos, ``Backpropagation through time: What it is and
how to do it,'' \emph{Proc. IEEE,} vol. 78, no. 10, pp. 1550--1560,
1990.

{[}22{]} W. T. Miller, R. S. Sutton, and P. J. Werbos, Eds.,
\emph{Neural Networks for} \emph{Control.} Cambridge, MA: MIT Press,
1990.

{[}23{]} D. A. White and D. A. Sofge, Eds., \emph{Handbook of
Intelligent Control:} \emph{Neural, Fuzzy, and Adaptive Approaches.} New
York: Van Nostrand Reinhold, 1992.
\end{quote}

{[}24{]} P. J. Werbos, ``Consistency of HDP applied to a simple
reinforcement learning problem,'' \emph{Neural Networks,} vol. 3, pp.
179--189, 1990.

\begin{quote}
{[}25{]} L. Baird, ``Residual algorithms: Reinforcement learning with
function approximation,'' in \emph{Proc. 12th Int. Conf. on Machine
Learning,} San Francisco, CA, July 1995, pp. 30--37.

{[}26{]} S. J. Bradtke, B. E. Ydstie, and A. G. Barto, ``Adaptive linear
quadratic control using policy iteration,'' in \emph{Proc. Amer. Contr.
Conf.,} Baltimore, MD, June 1994, pp. 3475--3479.
\end{quote}

{[}27{]} N. Borghese and M. Arbib, ``Generation of temporal sequences
using local dynamic programming,'' \emph{Neural Networks,} no. 1, pp.
39--54, 1995. {[}28{]} D. Prokhorov, ``A globalized dual heuristic
programming and its ap-plication to neurocontrol,'' in \emph{Proc. World
Congr. Neural Networks,} Washington, D.C., July 1995, pp. II-389--392.

\begin{quote}
{[}29{]} D. Prokhorov and D. Wunsch, ``Advanced adaptive critic
designs,'' in \emph{Proc. World Congress on Neural Networks,} San Diego,
CA, Sept. 1996, pp. 83--87.

{[}30{]} D. Prokhorov, R. Santiago, and D. Wunsch, ``Adaptive critic
designs: A case study for neurocontrol,'' \emph{Neural Networks,} vol.
8, no. 9, pp. 1367--1372, 1995.

{[}31{]} G. Puskorius and L. Feldkamp, ``Neurocontrol of nonlinear
dynamical systems with Kalman filter trained recurrent networks,''
\emph{IEEE Trans. Neural Networks,} vol. 5, pp. 279--297, 1994.

{[}32{]} G. Puskorius, L. Feldkamp, and L. Davis, ``Dynamic
neural-network methods applied to on-vehicle idle speed control,''
\emph{Proc. IEEE,} vol. 84, no. 10, pp. 1407--1420, 1996.

{[}33{]} F. Yuan, L. Feldkamp, G. Puskorius, and L. Davis, ``A simple
solution to the bioreactor benchmark problem by application of
Q-learning,'' in \emph{Proc. World Congr. Neural Networks,} Washington,
D.C., July 1995, pp. II-326--331.

{[}34{]} P. J. Werbos, ``Optimal neurocontrol: Practical benefits, new
results and biological evidence,'' in \emph{Proc. World Congr. Neural
Networks,} Washington, D.C., July 1995, pp. II-318--325.
\end{quote}

{[}35{]} R. Williams and D. Zipser, ``A learning algorithm for
continually running fully recurrent neural networks,'' \emph{Neural
Computa.,} vol. 1, pp. 270--280. {[}36{]} K. S. Narendra and K.
Parthasarathy, ``Identification and control of dy-namical systems using
neural networks,'' \emph{IEEE Trans. Neural Networks,} vol. 1, pp.
4--27.
\end{minipage} \\
\midrule()
\endhead
\bottomrule()
\end{longtable}

PROKHOROV AND WUNSCH: ADAPTIVE CRITIC DESIGNS 1007

\begin{quote}
{[}37{]} K. S. Narendra and A. M. Annaswamy, \emph{Stable Adaptive
Systems.} Englewood Cliffs, NJ: Prentice-Hall, 1989.

{[}38{]} R. Santiago and P. J. Werbos, ``A new progress toward truly
brain-like control,'' in \emph{Proc. World Congr. Neural Networks,} San
Diego, CA, June 1994, pp. I-27--33.

{[}39{]} L. Baird, ``Advantage updating,'' Wright Lab., Wright Patterson
AFB, Tech. Rep. WL-TR-93-1146, Nov. 1993.

{[}40{]} S. Thrun, \emph{Explanation-Based Neural Network Learning: A
Lifelong Learning Approach.} Boston, MA: Kluwer, 1996.

{[}41{]} H. White and A. Gallant, ``On learning the derivatives of an
unknown mapping with multilayer feedforward networks,'' \emph{Neural
Networks,} vol. 5, pp. 129--138, 1992.
\end{quote}

{[}42{]} D. Wunsch and D. Prokhorov, ``Adaptive critic designs,'' in
\emph{Computa-tional Intelligence: A Dynamic System Perspective,} R. J.
Marks, II, \emph{et} New York: IEEE Press, 1995, pp. 98--107.
\emph{al.,} Eds.

\begin{quote}
{[}43{]} S. N. Balakrishnan and V. Biega, ``Adaptive critic based neural
networks for control,'' in \emph{Proc. Amer. Contr. Conf.,} Seattle, WA,
June 1995, pp. 335--339.
\end{quote}

{[}44{]} P. Eaton, D. Prokhorov, and D. Wunsch, ``Neurocontrollers for
ball-and-beam systems,'' in \emph{Intelligent Engineering Systems
Through Artificial Neural Networks 6 (Proc. Conf. Artificial Neural
Networks in Engineer-}New York: Amer Soc. Mech. Eng. Press, \emph{ing),}
C. Dagli \emph{et al.,} Eds.

\begin{quote}
1996, pp. 551--557.

{[}45{]} K. S. Narendra and S. Mukhopadhyay, ``Adaptive control of
nonlinear multivariable systems using neural networks,'' \emph{Neural
Networks,} vol. 7, no. 5, pp. 737--752, 1994.

{[}46{]} N. Visnevski and D. Prokhorov, ``Control of a nonlinear
multivariable system with adaptive critic designs,'' in
\emph{Intelligent Engineering Systems Through Artificial Neural Networks
6 (Proc. Conf. Artificial Neural Networks in Engineering),} C. Dagli
\emph{et al.,} Eds. NY: Amer. Soc. Mech. Eng. Press, 1996, pp. 559--565;
note misprints in rms error values.

{[}47{]} K. KrishnaKumar, ``Adaptive critics: Theory and applications,''
tutorial at \emph{Conf. Artificial Neural Networks in Engineering
(ANNIE'96),} St. Louis, MO, Nov. 10--13, 1996.
\end{quote}

\textbf{Danil V. Prokhorov} (S'95) received the Honors

\includegraphics[width=1in,height=1.25in]{vertopal_328f8d44c10a4dcd8c6d79eba0b32a72/media/image17.png}
Texas Tech University, Lubbock, TX.

\begin{quote}
Automation of the Russian Academy of Sciences (formerly LIIAN), St.
Petersburg, Russia, as a Re-search Engineer. He worked at the Research
Labora-tory of Ford Motor Co., Dearborn, MI, as a Summer He worked at
the Institute for Informatics and Diploma in Robotics from the State
Academy of Aerospace Instrument Engineering (formerly LIAP), St.
Petersburg, Russia, in 1992. He is currently com-pleting the Ph.D.
degree in electrical engineering at
\end{quote}

Intern in 1995--1997. His research interests are in adaptive critics,
signal processing, system identification, control, and optimization
based on various neural networks.

Mr. Prokhorov is a member of the International Neural Network Society.

\textbf{Donald C. Wunsch, II} (SM'94) completed a Hu-

\begin{quote}
\includegraphics[width=1in,height=1.25in]{vertopal_328f8d44c10a4dcd8c6d79eba0b32a72/media/image18.png}mathematics
and the Ph.D. degree in electrical engi-neering from the University of
Washington, Seattle, in 1987 and 1991, respectively.

Seattle, WA, where he invented the first optical im-plementation of the
ART1 neural network, featured He was Senior Principal Scientist at
Boeing, manities Honors Program at Seattle University, WA, in 1981 and
received the B.S. degree in applied mathematics from the University of
New Mexico, Albuquerque, in 1984, the M.S. degree in applied

in the 1991 Annual Report, and other optical neural networks and applied
research contributions. He has also worked for International Laser
Systems and Rockwell International, both at Kirtland AFB, Albuquerque,
NM. He is Director of the Applied Computational Intelligence Laboratory
at Texas Tech University, Lubbock, TX, involving six other faculty,
several postdoctoral associates, doctoral candidates, and other graduate
and undergraduate students. His current research includes neural
optimization, forecasting, and control, financial engineering, fuzzy
risk assessment for high-consequence surety, wind engineering,
characterization of the cotton manufacturing process, intelligent
agents, and Go. He is heavily involved in research collaborations with
former Soviet scientists.

Dr. Wunsch is an Academician in the International Academy of
Tech-nological Cybernetics and the International Informatization
Academy. He is recipient of the Halliburton Award for excellence in
teaching and research at Texas Tech. He is a member of the International
Neural Network Society and a past member of the IEEE Neural Network
Council.
\end{quote}

\end{document}
