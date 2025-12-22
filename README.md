## Propeller and Wing Optimizer

#### Current Status
I have implemented multiprocessing for the optimization of airfoils, I am going to start creating a caller to run optimization at different conditions that the wing will experience in, and begin chaining together the different airfoils to create a wing. 

This will be a pretty quick fix, but the current optimization algorithm is creating airfoils that are physically impossible, but that's just because I haven't implemented the required constraints. AI wins again! Here is an example of a wing that is made to optimize lift (among other things, but mostly lift) 
![Really wonky wing](https://github.com/Licheng-Zheng/propellerandwingoptimizer/blob/10189384a29f27416ccd3d1377d2e7a127288202/media/wonkywing.png)
It should acheive pretty great lift, but is likely sacrificing just about everything else (very forward transition boundary (unstable, bad for efficiency), pretty high drag (to be expected because I specified a very low speed), literally not possible construction (pinches to 0)). But nothing that can't be fixed in one 240 hour lock-in session. All the issues aren't really issues with the methodology, but more because the stupid user did not implement enough constraints to make the wings relastic and viable. 

My Progress diagram, can't really show you it because then you can't see the entire thing and the image export is not very good :( Maybe I can add a video in the future....
![Progress Diagram](https://github.com/Licheng-Zheng/propellerandwingoptimizer/blob/e7c3d1151899df6fe70e1def9bad8f764783f5ca/media/progressdiagram.png)

After optimizing for a single airfoil, it also creates a 3d stl file of the wing, (which is just the airfoil extruded). It can be used for 3d printing (and machining I'd imagine), but it is still just the airfoil extuded so there's nothing too special about it. 
![media/Extruded Wing - No clue why its so long.png](https://github.com/Licheng-Zheng/propellerandwingoptimizer/blob/master/media/Extruded%20Wing%20-%20No%20clue%20why%20its%20so%20long.png))
![[Wing in Prusa slicer] (C:\Users\liche\OneDrive\Desktop\PycharmProjects\PropellerDesign\media\Wing in Prusaslicer again.png)](https://github.com/Licheng-Zheng/propellerandwingoptimizer/blob/master/media/Wing%20in%20Prusaslicer%20again.png)


#### Goal 
I want to provide the program a bunch of different parameters and conditions that the wing/propeller will be working within, and create the optimal wing/propeller design for those parameters. 

## Motivation
This started as an attempt to build a quiet, efficient propeller for a small air cooler for a telescope (then it got more and more interesting so I'm expanding the project now to include (hopefully) more functionality) The project explores whether automated optimization can produce high-efficiency propellers and wings.

### Installation Instructions
If you are an industry professional wanting to use this tool, please do not use this tool. If I have to get on a plane optimized with this program I am getting off the plane.
Not sure why anyone would want to use this thing, but if you do, here are the hopefully clear instructions:
1. Clone the repository from Github, you can use the following command: 'gh repo clone Licheng-Zheng/propellerandwingoptimizer' in the git bash
2. Navigate to the project directory, currently, only the 'usingasb' folder is complete and usable
3. Install the required dependencies, you can use the following command: 'pip install -r requirements.txt'
- You should probably create this under a virtual environment because it is a pretty large download. This gets you all the required libraries for the project. 
4. Go into main.py and run it, and that's it! It will use your local machine to optimize the airfoil. 
5. If you want to use modal for faster acceleration, go into the using_modal.py file in the root directory (I couldn't put it into the usingasb file because then the file searching system is a lot weirder), import modal in terminal, set an environment variable with your API key (or just set it in the file itself and make sure you don't push it to somewhere public) and tada! It should run. 


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
