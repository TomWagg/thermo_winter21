import numpy as np
import tkinter as tk
from itertools import combinations
from matplotlib.colors import rgb2hex
from matplotlib.pyplot import get_cmap
from scipy.stats import kstest


def speed_cdf(v, v_rms):
    return 1 - np.exp(-(v / v_rms)**2)


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

        if isinstance(masses, (int, float)):
            self.masses = np.repeat(masses, self.N)
        else:
            self.masses = masses

        self.radius = radius

        # convert initial kinetic energy into a velocity
        init_v = np.sqrt(2 * E / masses)

        # set random velocities for each particle (randomly distributed between x and y speed)
        vx = np.random.uniform(0, init_v, size=N) * np.random.choice([-1, 1], size=N)
        vy = np.sqrt(init_v**2 - vx**2) * np.random.choice([-1, 1], size=N)

        self.vel = np.transpose([vx, vy])
        self.visualise = visualise
        self.combos = np.array(list(combinations(np.arange(0, self.N).astype(int), 2)))

        if visualise:
            self.delay = delay
            self.canvas = None
            self.root = None
            self.particle_handles = {}

            self._init_visualization()
            self.root.update()

    def get_total_kinetic_energy(self):
        return 0.5 * np.sum(self.masses * np.sum(self.vel**2, axis=1))

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

        if np.all(self.masses == self.masses[0]):
            fill = np.repeat(rgb2hex(get_cmap("Greens")(0.5)), self.N)
        else:
            scaled_masses = (self.masses - self.masses.min()) / (self.masses.max() - self.masses.min())
            fill = [rgb2hex(get_cmap("viridis_r")(sm)) for sm in scaled_masses]

        reds = np.random.choice(self.N, size=5, replace=False)

        # add all of the particles
        for i in range(self.N):
            self.particle_handles[i] = self._draw_particle(i, fill=fill[i] if i not in reds else "red",
                                                           outline="black")

        # update this all on the canvas
        self.root.update()

    def _quit_visualisation(self):
        self.root.destroy()

    def _draw_particle(self, pid, fill="green", outline="black"):
        """Draw a circle on the canvas corresponding to particle

        Returns the handle of the tkinter circle element"""
        x0 = self.pos[pid, 0] - self.radius
        y0 = self.pos[pid, 1] - self.radius
        x1 = self.pos[pid, 0] + self.radius
        y1 = self.pos[pid, 1] + self.radius

        return self.canvas.create_oval(x0, y0, x1, y1, fill=fill, outline=outline)

    def reached_steadstate(self):
        self.speeds = np.sqrt(np.sum(self.vel**2, axis=1))
        v_rms = np.sqrt(np.mean(self.speeds**2))
        print(kstest(self.speeds, speed_cdf, args=(v_rms,)).statistic)
        return kstest(self.speeds, speed_cdf, args=(v_rms,)).statistic <= 0.03

    def resolve_particle_collisions(self):

        distances = np.sqrt(np.square(self.pos[self.combos][:, 0, :]
                                      - self.pos[self.combos][:, 1, :]).sum(axis=1))

        v_dotprod = np.sum(self.vel[self.combos][:, 0, :] * self.vel[self.combos][:, 1, :], axis=1)

        colliders = np.logical_and(distances <= 2 * self.radius, v_dotprod < 0)
        if len(colliders[colliders]) > 0:
            mask = self.combos[colliders]
            # print("COLLISION", self.combos[colliders])

            M = self.masses[mask].sum(axis=1)
            m1 = self.masses[mask][:, 0]
            m2 = self.masses[mask][:, 1]
            v1 = self.vel[mask][:, 0, :]
            v2 = self.vel[mask][:, 1, :]
            p1 = self.pos[mask][:, 0, :]
            p2 = self.pos[mask][:, 1, :]

            scalar_bit1 = 2 * m2 / M * np.sum((v1 - v2) * (p1 - p2), axis=1) / np.linalg.norm(p1 - p2, axis=1)**2
            scalar_bit2 = 2 * m1 / M * np.sum((v2 - v1) * (p2 - p1), axis=1) / np.linalg.norm(p2 - p1, axis=1)**2

            new_v1 = v1 - scalar_bit1[:, np.newaxis] * (p1 - p2)
            new_v2 = v2 - scalar_bit2[:, np.newaxis] * (p2 - p1)

            for i, combo in enumerate(self.combos[colliders]):
                self.vel[combo[0]] = new_v1[i]
                self.vel[combo[1]] = new_v2[i]

    def resolve_wall_collisions(self):
        """Reverse the direction of any particles that hit walls"""
        outside_x = np.logical_or(self.pos[:, 0] + self.radius >= self.size,
                                  self.pos[:, 0] - self.radius <= 0)
        outside_y = np.logical_or(self.pos[:, 1] + self.radius >= self.size,
                                  self.pos[:, 1] - self.radius <= 0)

        self.vel[:, 0][outside_x] = -self.vel[:, 0][outside_x]
        self.vel[:, 1][outside_y] = -self.vel[:, 1][outside_y]

    def run_simulation(self, steps=1000, run_until_steadstate=False):
        i = 0
        actual_steps = 0
        while i < steps:
            # 1. update all particle positions based on current speeds
            self.pos += self.vel

            if self.visualise:
                for j in range(self.N):
                    self.canvas.move(self.particle_handles[j], self.vel[j, 0], self.vel[j, 1])

            # 2. resolve whether any hit the wall and reflect them
            self.resolve_wall_collisions()

            # 3. resolve any particle collisions and transfer momentum
            self.resolve_particle_collisions()

            if self.visualise:
                # update visualization with a delay
                self.root.after(self.delay, self.root.update())

                # change the timestep message as well
                self.canvas.itemconfig(self.timestep_message, text="Timestep = {}".format(i))

            # only update i when we're stepping a set number of times
            if not run_until_steadstate:
                i += 1
            else:
                if self.reached_steadstate():
                    return actual_steps
                actual_steps += 1

        if self.visualise:
            self.root.mainloop()
