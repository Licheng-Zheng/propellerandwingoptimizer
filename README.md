### Propeller and Wing Optimizer

#### Current Status
I have implemented multiprocessing for the optimization of airfoils, I am going to start creating a caller to run optimization at different conditions that the wing will experience in, and begin chaining together the different airfoils to create a wing. 

#### Goal 
I want to provide the program a bunch of different parameters and conditions that the wing/propeller will be working within, and create the optimal wing/propeller design for those parameters. 

## Motivation
This started as an attempt to build a quiet, efficient propeller for a small air cooler for a telescope (then it got more and more interesting so I'm expanding the project now to include (hopefully) more functionality) The project explores whether automated optimization can produce high-efficiency propellers and wings.

## Roadmap
- [ ] Define input schema (operating conditions, constraints, objective).
- [ ] Baseline solver (e.g., lifting-line or panel method).
- [ ] Optimizer (start with gradient-free; benchmark vs. gradient-based).
- [ ] Validation against known airfoils/props.
- [ ] CLI and basic plots.

### Next Update Roadmap 
I'm going to delete this section when I'm complete the items in it, this is just so I know what I need to work on as well

#### main.py
- [ ] Dedicated class that runs the optimization, separating the optimization from the main file which currently has too many responsibilities (this would be a class with just the single cma run)
- [ ] Add a class that runs the multiprocessing (might not work if I want to have different optimizers (more than just a CMA-ES) because I'm not sure I would allocate resources in this case but I will see)
- [ ] Allow the user to load their own constants from their own file (saving path, importance lists) so its not just all clogging up my main file as it is currently, which is super ugly 
- [ ] No interactive blocking until I'm down the actual project because right now it is just making it harder to work on the real code

#### objective.py 
I think this one is more ok, but these are some recommendations from ChatGPT to increase the modularity of the code so I can swap things more easily in the future

I want to promote changing the scoring function, constraint suite (a different constraint suite is used if varying amounts of complexity checks are required (for example, easier and more basic checks at the start of the run to train faster))
- [ ] Split up the constraint evaluation from the actual NeuralFoil evaluation
- [ ] Give the constraints their own file

#### cma_optimization.py 
- [ ] The run_cma_optimization function doesn't run the cma_optimization, it only performs on step, so I need to refactor this (this is super easy to do but I just want an easy check mark ok 🥲)

#### logging_auxiliary_functions.py 
Honestly, get rid of this thing, blocking is really bad for multiprocessing 
- [ ] Move my interactive tools and plotting into a diagnostics module (have an interactive first, which is already implemented)

#### optimal_wing_state.py 
This is used for capturing the optimal wing state for future use 
- [ ] Split up all the functions of this file into different files (right now, the function mixes printing to console, writing file and everything)
- [ ] Create a presentation format into a module so the capture can be used without a GUI (without user interaction) 
