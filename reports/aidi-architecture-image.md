# AIDI architecture illustration

Generated with the built-in image_gen tool. Final asset: `docs/assets/images/aidi_architecture.png`.
The same English-labelled illustration is used in the English and Russian documentation.

## Generation prompt

```text
Use case: infographic-diagram.
Asset type: final raster illustration for the TensorAeroSpace documentation page about AIDI flight-control architecture.
Create a polished, technically accurate, publication-quality architecture infographic, landscape 3:2 composition, very sharp typography, generous whitespace, flat white background, clean rounded cards, restrained navy and blue with teal adaptation paths and an amber PCH card. This is a readable engineering figure, not a decorative poster. All words must be spelled correctly. Use large English labels, readable when the image is displayed at 700 pixels wide.

Title: "AIDI"
Subtitle: "Adaptive Incremental Dynamic Inversion"

Central main control flow, left to right, four aligned cards:
1. "Outer loop" with smaller "C* · roll · sideslip". Incoming label "Commands". Outgoing arrow label "Desired rates".
2. "Rate feedback" with formula "ν = Kω (ω_des − ω)". Outgoing arrow label "ν".
3. "Adaptive inversion" with formulas "G̃ = Θ ⊙ G_nominal" and "Δu = G̃⁺ (ν − ω̇_f)".
4. "Aircraft + actuators", with a small restrained aircraft line illustration and "env.step". Between card 3 and card 4 draw a clearly readable small connector block "Sum + limits" with "u_f + Δu", followed by arrow "u_cmd". Never send Δu straight to the aircraft.

Auxiliary cards in a clean second tier with routed arrows:
- Above Adaptive inversion: "Onboard model", small "G_nominal(x, u_applied)", output G_nominal enters Adaptive inversion.
- Below Aircraft: "Matched filters", small "Measured rates + actual controls". Aircraft outputs measured ω and u_applied to this card.
- Below Adaptive inversion: "ScalingRLS", small "Information-based VFF", second small line "Consistency check". A teal arrow runs from Matched filters to ScalingRLS labeled "Δω̇_f, Δu_f"; another teal arrow from ScalingRLS up to Adaptive inversion is labeled "Θ". Nominal control effectiveness is also an input to ScalingRLS; show a discreet label "with G_nominal" inside this card to avoid a long crossing arrow.
- Below Outer loop: amber "PCH" card with "h = ν_prev − ω̇_f". Its arrow returns upward to Outer loop labeled "h". Treat filtered measurements as a feedback bus below the main control cards; the bus supplies ω̇_f to inversion and PCH, u_f to Sum + limits, and ω to rate feedback. Use clearly separated paths and short signal tags, no crossing through boxes or labels. ν_prev for PCH is the previous rate-feedback demand.

Footer caption: "Measured feedback · Continuous identification · Actuator constraints"
Technical invariants: RLS adapts Θ, not the reference; the nominal model is multiplied elementwise by Θ; limits act on the sum of filtered actual position and the control increment; actual actuator feedback is used; do NOT show an active speed/throttle controller, a neural network, a reward block, or claim guaranteed recovery/stability.
Prioritize a coherent, uncluttered diagram with visible arrowheads, aligned blocks, consistent label hierarchy and plenty of margins. No watermarks, no photorealistic background, no fake 3D UI, no tiny footnotes.
```

## Correction prompt

```text
Edit this AIDI architecture infographic. Preserve the entire layout, all cards, typography, formulas, colors, aircraft drawing, and every other arrow exactly. Make ONLY these two technical corrections: (1) the vertical teal connection between the bottom of the 'Matched filters' card and the wide feedback bus beneath it must carry data DOWNWARD FROM Matched filters TO the feedback bus. Remove its upward-pointing arrowhead at the card and place a downward arrowhead at the bus, keeping the same connection location; all the other arrows from the bus up into the controller blocks must remain unchanged. (2) On the feedback bus replace the small text 'ω̇_f (filtered rates)' with 'ω̇_f (filtered acceleration)' exactly. Keep the other bus labels 'ω (measured rates)' and 'u_f (filtered actual controls)' unchanged. Do not change anything else and do not add any new words.
```
