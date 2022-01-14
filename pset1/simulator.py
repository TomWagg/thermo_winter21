import numpy as np
import tkinter as tk


class particle():
    def __init__(self, size, pid, init_ke=5, radius=3, mass=1):
        """Initialise the particles

        Parameters
        ----------
        size : int
            Size of the box
        pid : int
            Unique particle ID
        init_ke : int, optional
            Initial kinetic energy for the particle, by default 5
        radius : int, optional
            Radius of the particle, by default 3
        mass : int, optional
            Mass of the particle, by default 1
        """
        # choose random x and y positions within the grid (padded by radius of particles)
        self.pos = np.random.uniform(0 + radius, size - radius, size=2)

        # convert initial kinetic energy into a velocity
        init_v = np.sqrt(2 * init_ke / mass)

        # set random velocities for each particle (randomly distributed between x and y speed)
        self.vel = np.array([None, None])
        self.vel[0] = np.random.uniform(0, init_v) * np.random.choice([-1, 1])
        self.vel[1] = np.sqrt(init_v**2 - self.vel[0]**2) * np.random.choice([-1, 1])

        # set the radius and mass of the particle
        self.radius = radius
        self.mass = mass

        # assign a particle id to each particle
        self.pid = pid

    def update_pos(self, val):
        self.pos = val

    def update_vel(self, val):
        self.vel = val

    def update_vx(self, val):
        self.vel[0] = val

    def update_vy(self, val):
        self.vel[1] = val

    def colliding(self, other_particle):
        distance = np.sqrt(sum((other_particle.pos - self.pos)**2))
        return distance <= self.radius + other_particle.radius


class Simulation():  # this is where we will make them interact
    def __init__(self, N, E, size, radius, mass, delay=20, visualise=True):
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
        radius : `int`
            Radius of the particles
        mass : `int`
            Mass of the particles
        delay : `int`
            Delay in milliseconds between showing/running timesteps
        visualise : `boolean`
            Whether to animate the balls in the box
        """
        self.N = N
        self.E = E
        self.size = size

        # initialise N particle classes
        self.particles = [particle(size=size, pid=i, init_ke=E, radius=radius, mass=mass) for i in range(N)]
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
        for p in self.particles:
            self.particle_handles[p.pid] = self._draw_particle(p)

        # update this all on the canvas
        self.root.update()

    def _quit_visualisation(self):
        self.root.destroy()

    def _draw_particle(self, particle):
        """Draw a circle on the canvas corresponding to particle

        Returns the handle of the tkinter circle element"""
        x0 = particle.pos[0] - particle.radius
        y0 = particle.pos[1] - particle.radius
        x1 = particle.pos[0] + particle.radius
        y1 = particle.pos[1] + particle.radius

        colours = ["black", "red", "blue", "green"]

        return self.canvas.create_oval(x0, y0, x1, y1, fill=np.random.choice(colours), outline='black')

    def _move_particle(self, particle):
        new_pos = particle.pos + particle.vel
        particle.update_pos(new_pos)

        if self.visualise:
            self.canvas.move(self.particle_handles[particle.pid], particle.vel[0], particle.vel[1])

    def resolve_particle_collisions(self):
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
                    print(p1.pid, p2.pid, "are colliding")
                    not_yet_collided.discard(p2)

                    M = p1.mass + p2.mass

                    new_v1 = p1.vel - 2 * p2.mass / M * np.dot(p1.vel - p2.vel, p1.pos - p2.pos) / np.linalg.norm(p1.pos - p2.pos)**2 * (p1.pos - p2.pos)
                    new_v2 = p2.vel - 2 * p1.mass / M * np.dot(p2.vel - p1.vel, p2.pos - p1.pos) / np.linalg.norm(p2.pos - p1.pos)**2 * (p2.pos - p1.pos)

                    p1.update_vel(new_v1)
                    p2.update_vel(new_v2)
                    break

    def resolve_wall_collisions(self):
        """Reverse the direction of any particles that hit walls"""
        for particle in self.particles:
            if (particle.pos[0] + particle.radius) >= self.size or (particle.pos[0] - particle.radius) <= 0:
                particle.update_vx(-particle.vel[0])

            if (particle.pos[1] + particle.radius) >= self.size or (particle.pos[1] - particle.radius) <= 0:
                particle.update_vy(-particle.vel[1])

    def run_simulation(self, steps=1000):
        for i in range(steps):
            # 1. update all particle positions based on current speeds
            for particle in self.particles:
                self._move_particle(particle)

            # 2. resolve whether any hit the wall and reflect them
            self.resolve_wall_collisions()

            # 3. resolve any particle collisions and transfer momentum
            self.resolve_particle_collisions()

            if self.visualise:
                # update visualization with a delay
                self.root.after(self.delay, self.root.update())

                # change the timestep message as well
                self.canvas.itemconfig(self.timestep_message, text="Timestep = {}".format(i))
        
        if self.visualise:
            self.root.mainloop()

    def get_velocities(self):
        raise NotImplementedError
