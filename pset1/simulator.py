import numpy as np
import tkinter as tk


class Simulation():  # this is where we will make them interact
    def __init__(self, N, E, size, radius, masses, delay=20, visualise=True):
        """Simulation class initialisation. This class handles the entire particle
        in a box thing.

        Parameters
        ----------
        N : `int`
            Total number of particles
        E : `int`
            Kinetic energy to start with
        size : `int`
            Size of the box
        radius : `int` or `list`
            Radius of the particles
        masses : `int` or `list`
            Mass of the particles
        delay : `int`
            Delay in milliseconds between showing/running timesteps
        visualise : `boolean`
            Whether to animate the balls in the box
        """
        self.N = N
        self.E = E
        self.size = size

        # create an empty position array and try to find a position for each particle
        positions = []
        for _ in range(N):
            # keep looping until the position is no longer invalid
            position_invalid = True
            while position_invalid:
                # pick a random (x, y) position
                possible_pos = np.random.uniform(0 + radius, size - radius, size=2)
                position_invalid = False

                # loop over all other chosen positions
                for other in positions:
                    # mark the position as bad if it overlaps with another particle
                    position_invalid = np.sqrt(sum((other - possible_pos)**2)) <= 2 * radius
                    if position_invalid:
                        break

            # add to the position array
            positions.append(possible_pos)

        self.pos = np.array(positions)
        self.masses = masses
        self.radius = radius

        # convert initial kinetic energy into a velocity
        init_v = np.sqrt(2 * E / masses)

        # set random velocities for each particle (randomly distributed between x and y speed)
        vx = np.random.uniform(0, init_v, size=N) * np.random.choice([-1, 1], size=N)
        vy = np.sqrt(init_v**2 - vx**2) * np.random.choice([-1, 1], size=N)

        self.vel = np.transpose([vx, vy])
        self.visualise = visualise

        if visualise:
            self.delay = delay
            self.canvas = None
            self.root = None
            self.particle_handles = {}

            self._init_visualization()
            self.root.update()

    def _init_visualization(self):
        # start the visualisation box
        self.root = tk.Tk()
        self.root.title("Particles in a Box!")

        # create a canvas with the right size
        self.canvas = tk.Canvas(self.root, width=self.size, height=self.size)
        self.canvas.pack()

        # add a close button
        self.button = tk.Button(self.root, text='Close', command=self._quit_visualisation)
        self.button.place(x=self.size, y=10, anchor="e")

        self.timestep_message = self.canvas.create_text(self.size // 2, 10, text="Timestep = 0")

        # add all of the particles
        for i in range(self.N):
            self.particle_handles[i] = self._draw_particle(i)

        # update this all on the canvas
        self.root.update()

    def _quit_visualisation(self):
        self.root.destroy()

    def _draw_particle(self, pid):
        """Draw a circle on the canvas corresponding to particle

        Returns the handle of the tkinter circle element"""
        x0 = self.pos[pid, 0] - self.radius
        y0 = self.pos[pid, 1] - self.radius
        x1 = self.pos[pid, 0] + self.radius
        y1 = self.pos[pid, 1] + self.radius

        colours = ["black", "red", "blue", "green"]

        return self.canvas.create_oval(x0, y0, x1, y1, fill=np.random.choice(colours), outline='black')

    def resolve_particle_collisions(self):

        def colliding(self, other_particle):
            distance = np.sqrt(sum((other_particle.pos - self.pos)**2))
            return distance <= self.radius + other_particle.radius
        # make a set of particles that haven't collided yet
        not_yet_collided = set(self.particles[:])

        # go through every single particle
        for p1 in self.particles:
            # we're handling its collisions now so remove it from the set
            not_yet_collided.discard(p1)

            # go through all potential colliders and check if they are colliding
            for p2 in list(not_yet_collided):
                if p1.colliding(p2):
                    # handle the collision!
                    not_yet_collided.discard(p2)

                    M = p1.mass + p2.mass

                    new_v1 = p1.vel - 2 * p2.mass / M * np.dot(p1.vel - p2.vel, p1.pos - p2.pos) / np.linalg.norm(p1.pos - p2.pos)**2 * (p1.pos - p2.pos)
                    new_v2 = p2.vel - 2 * p1.mass / M * np.dot(p2.vel - p1.vel, p2.pos - p1.pos) / np.linalg.norm(p2.pos - p1.pos)**2 * (p2.pos - p1.pos)

                    p1.update_vel(new_v1)
                    p2.update_vel(new_v2)
                    break

    def resolve_wall_collisions(self):
        """Reverse the direction of any particles that hit walls"""
        outside_x = np.logical_or(self.pos[:, 0] + self.radius >= self.size,
                                  self.pos[:, 0] - self.radius <= 0)
        outside_y = np.logical_or(self.pos[:, 1] + self.radius >= self.size,
                                  self.pos[:, 1] - self.radius <= 0)

        self.vel[:, 0][outside_x] = -self.vel[:, 0][outside_x]
        self.vel[:, 1][outside_y] = -self.vel[:, 1][outside_y]

    def run_simulation(self, steps=1000):
        for i in range(steps):
            # 1. update all particle positions based on current speeds
            self.pos += self.vel

            if self.visualise:
                for j in range(self.N):
                    self.canvas.move(self.particle_handles[j], self.vel[j, 0], self.vel[j, 1])

            # 2. resolve whether any hit the wall and reflect them
            self.resolve_wall_collisions()

            # 3. resolve any particle collisions and transfer momentum
            # self.resolve_particle_collisions()

            if self.visualise:
                # update visualization with a delay
                self.root.after(self.delay, self.root.update())

                # change the timestep message as well
                self.canvas.itemconfig(self.timestep_message, text="Timestep = {}".format(i))

        if self.visualise:
            self.root.mainloop()
