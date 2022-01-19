import numpy as np
import tkinter as tk
from itertools import combinations
from matplotlib.colors import rgb2hex
from matplotlib.pyplot import get_cmap
from scipy.stats import kstest


def speed_cdf(v, v_rms):
    """Cumulative distribution function for the speeds

    Parameters
    ----------
    v : `float/array`
        Speeds at which to evaluate
    v_rms : `float`
        The root mean square speed of the distribution
    """
    return 1 - np.exp(-(v / v_rms)**2)


def energy_cdf(E, m, v_rms):
    """Cumulative distribution function for the energies

    Parameters
    ----------
    E : `float/array`
        Energies at which to evaluate
    m : `float`
        Masses of the particles (this currently only handles particles with the same mass)
    v_rms : `float`
        The root mean square speed of the distribution
    """
    kBT = 0.5 * m * v_rms**2
    return 1 - np.exp(-E / kBT)


class Simulation():
    def __init__(self, N, E, size, radius, masses, delay=20, visualise=True):
        """Simulation class initialisation. This class handles the entire particle in a box situation.

        Parameters
        ----------
        N : `int`
            Total number of particles
        E : `float`
            Kinetic energy for each particle to start with
        size : `float`
            Size of the box
        radius : `float/array`
            Radius of the particles
        masses : `float/array`
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
        self.radius = radius

        # save the masses as an array, whatever is inputted
        if isinstance(masses, (int, float)):
            self.masses = np.repeat(masses, self.N)
        else:
            self.masses = masses

        # convert initial kinetic energy into a velocity
        init_v = np.sqrt(2 * E / masses)

        # set random velocities for each particle (randomly distributed between x and y speed)
        vx = np.random.uniform(0, init_v, size=N) * np.random.choice([-1, 1], size=N)
        vy = np.sqrt(init_v**2 - vx**2) * np.random.choice([-1, 1], size=N)
        self.vel = np.transpose([vx, vy])

        # this constant defines the unique combinations of particles so I don't recompute it every time
        self.combos = np.array(list(combinations(np.arange(0, self.N).astype(int), 2)))

        # initialise visualisation if it is turned on
        self.visualise = visualise
        if visualise:
            self.delay = delay
            self.canvas = None
            self.root = None
            self.particle_handles = {}

            self._init_visualization()
            self.root.update()

    def _init_visualization(self):
        """ Start up the visualisation stuff and save it all in the class """
        # start the visualisation box
        self.root = tk.Tk()
        self.root.title("Particles in a Box!")

        # create a canvas with the right size
        self.canvas = tk.Canvas(self.root, width=self.size, height=self.size)
        self.canvas.pack()

        # add a close button
        self.button = tk.Button(self.root, text='Close', command=self._quit_visualisation)
        self.button.place(x=self.size, y=10, anchor="e")

        # add a message that keeps track of the timesteps
        self.timestep_message = self.canvas.create_text(self.size // 2, 10, text="Timestep = 0")

        if np.all(self.masses == self.masses[0]):
            # if all of the masses are the same then use green balls
            fill = np.repeat(rgb2hex(get_cmap("Greens")(0.5)), self.N)
        else:
            # otherwise make the colours represent the masses
            scaled_masses = (self.masses - self.masses.min()) / (self.masses.max() - self.masses.min())
            fill = [rgb2hex(get_cmap("viridis_r")(sm)) for sm in scaled_masses]

        # choose five random particles to make red
        reds = np.random.choice(self.N, size=5, replace=False)

        # add all of the particles
        for i in range(self.N):
            self._draw_particle(i, fill=fill[i] if i not in reds else "red", outline="black")

        # update this all on the canvas
        self.root.update()

    def _quit_visualisation(self):
        """ End the visualisation and close the canvas """
        self.root.destroy()

    def _draw_particle(self, pid, fill="green", outline="black"):
        """Draw a circle on the canvas representing the particle

        Parameters
        ----------
        pid : `int`
            The particle ID
        fill : `str`, optional
            Particle fill colour, by default "green"
        outline : str, optional
            Particle outline colour, by default "black"
        """
        x0 = self.pos[pid, 0] - self.radius
        y0 = self.pos[pid, 1] - self.radius
        x1 = self.pos[pid, 0] + self.radius
        y1 = self.pos[pid, 1] + self.radius

        self.particle_handles[pid] = self.canvas.create_oval(x0, y0, x1, y1, fill=fill, outline=outline)

    def reached_steadstate(self):
        """ Assess whether the simulation has reached a steady state. NOTE: this assumes equal masses """
        # calculate the speeds and root-mean-square speed
        self.speeds = np.sqrt(np.sum(self.vel**2, axis=1))
        self.energies = 0.5 * self.masses * self.speeds**2
        v_rms = np.sqrt(np.mean(self.speeds**2))

        # perform K-S tests on both the speeds and energies
        passes_speed = kstest(self.speeds, speed_cdf, args=(v_rms,)).pvalue >= 0.05
        passes_energy = kstest(self.energies, energy_cdf, args=(self.masses[0], v_rms,)).pvalue >= 0.05

        # only steady-state once both pass
        return passes_speed and passes_energy

    def resolve_particle_collisions(self):
        """ Resolve all particles collisions during this timestep """
        # calculate the distance between all pairs of particles and the dot product of their velocities too
        distances = np.sqrt(np.square(self.pos[self.combos][:, 0, :]
                                      - self.pos[self.combos][:, 1, :]).sum(axis=1))
        v_dotprod = np.sum(self.vel[self.combos][:, 0, :] * self.vel[self.combos][:, 1, :], axis=1)

        # define collisions as when distances are less than twice the radius and moving towards each other
        colliders = np.logical_and(distances <= 2 * self.radius, v_dotprod < 0)

        # if there are colliders
        if len(colliders[colliders]) > 0:
            mask = self.combos[colliders]

            # define a bunch of variables for clarity
            M = self.masses[mask].sum(axis=1)
            m1 = self.masses[mask][:, 0]
            m2 = self.masses[mask][:, 1]
            v1 = self.vel[mask][:, 0, :]
            v2 = self.vel[mask][:, 1, :]
            p1 = self.pos[mask][:, 0, :]
            p2 = self.pos[mask][:, 1, :]

            # compute the scalar parts of the this first
            scalar1 = 2 * m2 / M * np.sum((v1 - v2) * (p1 - p2), axis=1) / np.linalg.norm(p1 - p2, axis=1)**2
            scalar2 = 2 * m1 / M * np.sum((v2 - v1) * (p2 - p1), axis=1) / np.linalg.norm(p2 - p1, axis=1)**2

            # then compute the vectors (broadcasting the scalars upwards)
            new_v1 = v1 - scalar1[:, np.newaxis] * (p1 - p2)
            new_v2 = v2 - scalar2[:, np.newaxis] * (p2 - p1)

            # update the velocity of each combination sequentially
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

    def run_simulation(self, seconds=1000, run_until_steadstate=False):
        """Run the simulation of particles! It can either be run for a set amount of time or until a steady
        state is reached.

        Parameters
        ----------
        seconds : `int`, optional
            How many seconds to evolve for (ignored if `run_until_steady_state=True`), by default 1000
        run_until_steadstate : `bool`, optional
            Whether to run until steady state, by default False

        Returns
        -------
        t_relax : `int`
            Relaxation time (only returned when `run_until_steady_state=True`)
        """
        time = 0
        t_relax = 0
        while time < seconds:
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
                self.canvas.itemconfig(self.timestep_message, text="Timestep = {}".format(time))

            # only update time when we're stepping a set number of times
            if not run_until_steadstate:
                time += 1
            else:
                # check if we've reached steady state and return if so
                if self.reached_steadstate():
                    return t_relax
                # otherwise update the relaxation time
                t_relax += 1

        # if visualising then block until the canvas is closed by the user
        if self.visualise:
            self.root.mainloop()
