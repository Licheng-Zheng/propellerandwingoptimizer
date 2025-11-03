## Propeller and Wing Optimizer

#### Current Status
I have implemented multiprocessing for the optimization of airfoils, I am going to start creating a caller to run optimization at different conditions that the wing will experience in, and begin chaining together the different airfoils to create a wing. 

#### Goal 
I want to provide the program a bunch of different parameters and conditions that the wing/propeller will be working within, and create the optimal wing/propeller design for those parameters. 

## Motivation
This started as an attempt to build a quiet, efficient propeller for a small air cooler for a telescope (then it got more and more interesting so I'm expanding the project now to include (hopefully) more functionality) The project explores whether automated optimization can produce high-efficiency propellers and wings.
### Project Design 
**Input Formatting**
Wing 
- The speed in which the wing will be operating at (airspeed)
- Number of cuts throughout the wing, a separate airfoil will be used at each stage and different airfoils will be mixed and matched to test performance with each other
- Maximum parameters of the wing (height, length, thickness, maybe weight if one day the knowledge of mankind is deposited into my brain)
- The parameters to optimize for (the wanted and importance lists if you looked through my code. This tells the program that I want to optimize for lift, or thrust or reduce drag, very depending on your purposes)

Propeller 
Many of the same parameters carry over, but I haven't even started on the propeller portion of the project, so much of this is just what I think will be needed, will update when I get going
- RPM of the motor hub, this is important in figuring out the airspeed for each airfoil section of the wign
- Diameter of the propeller
- Number of props on the propeller

**Evaluation**
Airfoils - This is performed by a neural network called NeuralFoil, which takes in the cst parameters of your wing and predicts the lift, thrust and other parameters that might be expected of the wing.
Wing and Propeller - I'm going to try to create a BEMT (I think it stands for blade element momentum theory? Or something like that) to calculate the combined performance of the entire object 


### Roadmap
- [ ] Define input schema (create a template for the input that will be passed into the program)
- [ ] Baseline solver (try to create a neural network that can predict the best first guess of the required conditions, will require me to create my own data base of different NACA airfoils) 
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

### Things I'm going to try to do 
Currently with multiprocessing, my computer can run 4 separate instances of the program (optimizing 4 separate airfoils at the same time) (I have more cores than that so I could squeeze it out a bit faster, but I like to use my computer while its doing its thing). It has an AMD GPU, so if I wanted to speed it up with my GPU, I would need to figure out a way to get everything to work with Open CL, which seems like a lot of work to figure out. 
- Nvidia GPU
- More cores
These are the two options that I think will be more feasible for me to try to get things to speed up when I have finalized all my code. I will need to figure out how to use Nvidia GPUs, but I think they will be able to do everything much much faster. On the other hand, more cores would require basically no code changes for a pretty large speed up (worst case scenario I just let my program run for 10 days and hopefully it cooks up a nice wing by the time I'm done) 
