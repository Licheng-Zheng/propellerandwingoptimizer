### Propeller and Wing Optimizer

#### Goal 
I want to provide the program a bunch of different parameters and conditions that the wing/propeller will be working within, and create the optimal wing/propeller design for those parameters. 

## Motivation
This started as an attempt to build a quiet, efficient propeller for a small personal air conditioner. The project explores whether automated optimization can produce high-efficiency propellers and wings.#### So, uh, when you gonna fix this thing
When I have more than 2 brain cells to rub together and this becomes a real project I'll write more stuff here, until someone wants to use this and they fork it (I will notice in 1 to 3 business months), this shall remain a dumpster fire. 

Absolutely trolling, I had to rename my local branch to main so I wouldn't have some random branch named main, but too late now it is what it is. I also messed up in making my gitignore thing so now you can see all my monkey code.

## Roadmap
- [ ] Define input schema (operating conditions, constraints, objective).
- [ ] Baseline solver (e.g., lifting-line or panel method).
- [ ] Optimizer (start with gradient-free; benchmark vs. gradient-based).
- [ ] Validation against known airfoils/props.
- [ ] CLI and basic plots.

## Contributing notes
Development/branch conventions and .gitignore details will live in CONTRIBUTING.md. If the default branch or ignore rules change, they’ll be documented there.