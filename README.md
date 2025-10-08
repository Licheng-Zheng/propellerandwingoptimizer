## Propeller and Wing Optimizer

#### Goal 
I want to provide the program a bunch of different parameters and conditions that the wing/propeller will be working within, and create the optimal wing/propeller design for those parameters. 

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

### Things I'm going to try to do 
Currently with multiprocessing, my computer can run 4 separate instances of the program (optimizing 4 separate airfoils at the same time) (I have more cores than that so I could squeeze it out a bit faster, but I like to use my computer while its doing its thing). It has an AMD GPU, so if I wanted to speed it up with my GPU, I would need to figure out a way to get everything to work with Open CL, which seems like a lot of work to figure out. 
- Nvidia GPU
- More cores
These are the two options that I think will be more feasible for me to try to get things to speed up when I have finalized all my code. I will need to figure out how to use Nvidia GPUs, but I think they will be able to do everything much much faster. On the other hand, more cores would require basically no code changes for a pretty large speed up (worst case scenario I just let my program run for 10 days and hopefully it cooks up a nice wing by the time I'm done) 
