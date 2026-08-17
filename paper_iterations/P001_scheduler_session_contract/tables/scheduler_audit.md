# Table P001-A. Diffusion state-time contract

| schedule | initial state | first query | scheduler stride | actual next state | next query label | RMS to declared next state | student / teacher queries |
|---|---:|---:|---:|---:|---:|---:|---:|
| Historical modified-RAP heuristic | 8 | 10 | 1 | 9 | 0 | 0.0198078 | 2 / 2 |
| Scheduler-consistent Drive-OPD | 10 | 10 | 10 | 0 | 0 | 1.538e-8 | 2 / 2 |

The numerical probe uses the same clean trajectories, Gaussian noise, scaled-linear beta schedule,
`prediction_type=sample`, and Diffusers 0.21.4. It audits time labels and computation count, not
closed-loop driving quality.
