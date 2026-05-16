# Exercise 03: Learning-Rate Scheduler

Build any small model and an `Adam` optimizer with `lr=0.1`. Wrap the optimizer
in a `StepLR(step_size=5, gamma=0.5)` scheduler.

Loop 20 "epochs". After each one call `scheduler.step()` and print
`opt.param_groups[0]["lr"]`. The lr should halve every 5 epochs.
