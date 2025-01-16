import numpy as np
import soundcard as sc
import pygame
from scipy.fft import fft
import time
from scipy.spatial.distance import pdist, squareform
from collections import deque

class AudioVisualizer:
    def __init__(self):
        pygame.init()
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("AVOsmosis_0")
        
        # Audio setup
        self.SAMPLE_RATE = 44100
        self.CHANNELS = 2
        self.CHUNK = 1024
        self.speakers = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)
        
        # Signal history for relative changes
        self.signal_history = np.zeros(50)  # Store recent signal levels
        self.frequency_history = np.zeros((50, 3))  # Store bass, mids, highs history
        self.running_mean = 0
        self.running_std = 1
        self.history_alpha = 0.05  # Control rate of history update
        
        # Node system parameters
        self.num_nodes = 300
        self.nodes = np.random.rand(self.num_nodes, 2) * [self.width, self.height]
        self.velocities = np.zeros((self.num_nodes, 2))
        self.accelerations = np.zeros((self.num_nodes, 2))
        
        # Enhanced disruption parameters
        self.disruption_cooldown = 0
        self.peak_threshold = 0.3  # Lower threshold for more frequent disruptions
        self.peak_history = np.zeros(20)  # Longer history for better detection
        
        # Physics parameters
        self.connection_threshold = 400
        self.repulsion_strength = 2500
        self.attraction_strength = 200
        self.velocity_cap = 30.0  # Increased for more dramatic movements
        self.min_distance = 300.0
        self.max_connections = 12
        
        # Color parameters
        self.node_colors = np.zeros((self.num_nodes, 3))
        self.color_modes = np.random.randint(0, 3, self.num_nodes)  # Assign different color modes to nodes
        self.node_base_size = 2
        
    def process_audio(self, data):
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        # Normalize input data
        data = data / (np.max(np.abs(data)) + 1e-6)
        
        # Update running statistics
        current_mean = np.mean(np.abs(data))
        self.running_mean = (1 - self.history_alpha) * self.running_mean + self.history_alpha * current_mean
        self.running_std = (1 - self.history_alpha) * self.running_std + self.history_alpha * np.std(data)
        
        # Calculate FFT
        fft_data = fft(data)
        fft_freq = np.abs(fft_data[:len(fft_data)//2])
        fft_freq = np.log10(fft_freq + 1)
        
        # Normalize frequency bands relative to history
        freq_bands = np.array([
            np.mean(fft_freq[:int(len(fft_freq)*0.08)]),  # bass
            np.mean(fft_freq[int(len(fft_freq)*0.08):int(len(fft_freq)*0.4)]),  # mids
            np.mean(fft_freq[int(len(fft_freq)*0.4):])  # highs
        ])
        
        # Update frequency history
        self.frequency_history = np.roll(self.frequency_history, 1, axis=0)
        self.frequency_history[0] = freq_bands
        
        # Calculate relative changes
        freq_means = np.mean(self.frequency_history, axis=0)
        freq_stds = np.std(self.frequency_history, axis=0)
        relative_bands = (freq_bands - freq_means) / (freq_stds + 1e-6)
        
        # Scale and clip the relative changes
        bass = np.clip((relative_bands[0] + 3) / 6, 0, 1)
        mids = np.clip((relative_bands[1] + 3) / 6, 0, 1)
        highs = np.clip((relative_bands[2] + 3) / 6, 0, 1)
        
        # Enhanced peak detection using relative changes
        current_peak = np.max(np.abs(relative_bands))
        self.peak_history = np.roll(self.peak_history, 1)
        self.peak_history[0] = current_peak
        
        # Detect significant relative changes
        is_peak = (current_peak > self.peak_threshold and 
                  current_peak > 1.5 * np.mean(self.peak_history[1:]))
        
        return bass, mids, highs, is_peak, relative_bands
    
    def apply_disruption(self, strength, relative_bands):
        # Create multiple centers of disruption
        num_centers = 3
        centers = np.random.rand(num_centers, 2) * [self.width, self.height]
        
        for center in centers:
            directions = self.nodes - center
            distances = np.linalg.norm(directions, axis=1)
            normalized_directions = directions / (distances[:, np.newaxis] + 1e-6)
            
            # Add rotational component with varying strength
            perpendicular = np.column_stack((-normalized_directions[:, 1], 
                                           normalized_directions[:, 0]))
            
            # Scale disruption by relative frequency band strengths
            radial_strength = strength * (1 + np.abs(relative_bands[0]))  # Bass affects radial
            rot_strength = strength * (1 + np.abs(relative_bands[1]))     # Mids affect rotation
            
            disruption = (normalized_directions * radial_strength + 
                         perpendicular * rot_strength) / num_centers
            
            # Apply disruption to velocities with distance falloff
            falloff = 1 / (1 + distances/300)
            self.velocities += disruption * falloff[:, np.newaxis]
    
    def update_node_colors(self, bass, mids, highs, speeds):
        # Create different color schemes based on node modes
        for i in range(self.num_nodes):
            if self.color_modes[i] == 0:
                # Heat-based colors
                self.node_colors[i] = np.array([
                    30 + speeds[i] * 225 + bass * 155,
                    30 + mids * 225,
                    30 + highs * 155
                ])
            elif self.color_modes[i] == 1:
                # Cool colors
                self.node_colors[i] = np.array([
                    30 + highs * 155,
                    30 + speeds[i] * 225,
                    30 + bass * 225 + mids * 155
                ])
            else:
                # Purple-green spectrum
                self.node_colors[i] = np.array([
                    30 + mids * 225 + highs * 155,
                    30 + bass * 225,
                    30 + speeds[i] * 225 + mids * 155
                ])
            
            # Ensure colors stay within valid range
            self.node_colors[i] = np.clip(self.node_colors[i], 30, 255)
    
    def update_nodes(self, bass, mids, highs, is_peak, relative_bands):
        distances = squareform(pdist(self.nodes))
        forces = self.calculate_forces(distances, bass, mids)
        
        # Enhanced disruption handling
        if is_peak and self.disruption_cooldown <= 0:
            disruption_strength = 300.0 * (1 + np.max(np.abs(relative_bands)))
            self.apply_disruption(disruption_strength, relative_bands)
            self.disruption_cooldown = 3  # Shorter cooldown for more frequent disruptions
        else:
            self.disruption_cooldown = max(0, self.disruption_cooldown - 1)
        
        # Add random impulses based on relative changes
        random_impulses = np.random.randn(self.num_nodes, 2) * (
            np.abs(relative_bands[0]) * 8.0 + 
            np.abs(relative_bands[1]) * 4.0 +
            np.abs(relative_bands[2]) * 2.0
        )
        self.velocities += random_impulses
        
        # Update motion
        net_forces = np.sum(forces, axis=1)
        self.accelerations = net_forces * (1 + np.max(np.abs(relative_bands)) * 2.0)
        self.velocities = self.velocities * 0.92 + self.accelerations * 0.18
        
        # Dynamic velocity cap based on audio intensity
        current_cap = self.velocity_cap * (1 + np.max(np.abs(relative_bands)) * 2)
        speed = np.linalg.norm(self.velocities, axis=1)
        mask = speed > current_cap
        self.velocities[mask] *= current_cap / speed[mask, np.newaxis]
        
        # Update positions
        self.nodes += self.velocities
        
        # Boundary handling
        for i in range(2):
            mask_min = self.nodes[:, i] < 0
            mask_max = self.nodes[:, i] > [self.width, self.height][i]
            self.velocities[mask_min | mask_max, i] *= -0.8
            self.nodes[:, i] = np.clip(self.nodes[:, i], 0, [self.width, self.height][i])
        
        # Update colors with relative motion
        speeds = np.linalg.norm(self.velocities, axis=1) / current_cap
        self.update_node_colors(bass, mids, highs, speeds)
        
    def calculate_forces(self, distances, bass, mids):
        forces = np.zeros((self.num_nodes, self.num_nodes, 2))
        pos_diff = self.nodes[:, np.newaxis] - self.nodes
        
        np.fill_diagonal(distances, self.min_distance)
        directions = pos_diff / (distances[..., np.newaxis] + 1e-6)
        
        force_magnitudes = np.where(
            distances < self.connection_threshold,
            -self.attraction_strength * np.exp(-distances/50) * (1 + bass * 2),
            self.repulsion_strength / (distances ** 1.5) * (1 + mids)
        )
        
        forces = directions * force_magnitudes[..., np.newaxis] * 0.1
        chaos_term = np.random.randn(*forces.shape) * mids * 25
        forces += chaos_term
        
        return forces
    
    def draw_nodes_and_edges(self, bass, mids, highs):
        distances = squareform(pdist(self.nodes))
        current_threshold = self.connection_threshold * (1 + bass)
        
        # Draw edges
        for i in range(self.num_nodes):
            connections = np.where(distances[i] < current_threshold)[0]
            max_conn = int(self.max_connections * (1 + mids))
            
            if len(connections) > max_conn:
                connections = connections[:max_conn]
            
            for j in connections:
                if j > i:
                    intensity = 1 - (distances[i][j] / current_threshold)
                    color = (int(self.node_colors[i][0] * intensity),
                           int(self.node_colors[i][1] * intensity),
                           int(self.node_colors[i][2] * intensity))
                    
                    pygame.draw.line(self.screen, color,
                                   self.nodes[i], self.nodes[j],
                                   max(1, int(intensity * 2)))
        
        # Draw nodes with capped sizes
        max_node_size = 6  # Set maximum node size here
        speeds = np.linalg.norm(self.velocities, axis=1) / self.velocity_cap
        for i in range(self.num_nodes):
            color = tuple(map(int, self.node_colors[i]))
            size = max(self.node_base_size,
                      min(max_node_size,  # Add size cap
                          int(self.node_base_size * (1 + speeds[i] + highs * 2))))
            pygame.draw.circle(self.screen, color, self.nodes[i].astype(int), size)
    
    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        with self.speakers.recorder(samplerate=self.SAMPLE_RATE) as mic:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            running = False
                
                data = mic.record(numframes=self.CHUNK)
                bass, mids, highs, is_peak, relative_bands = self.process_audio(data)
                
                self.update_nodes(bass, mids, highs, is_peak, relative_bands)
                
                self.screen.fill((0, 0, 0, 10))
                self.draw_nodes_and_edges(bass, mids, highs)
                
                pygame.display.flip()
                clock.tick(60)
        
        pygame.quit()


class EnhancedAudioVisualizer(AudioVisualizer):
    def __init__(self):
        super().__init__()
        
        # Spectral flow visualization
        self.spectrum_history = deque(maxlen=50)
        self.flow_surface = pygame.Surface((self.width//4, self.height//3))
        self.flow_surface.set_alpha(180)
        
        # Beat detection
        self.energy_window = deque(maxlen=43)
        self.beat_threshold = 1.3
        self.last_beat = 0
        self.min_beat_interval = 10
        
        # Particle system
        self.particles = []
        self.particle_lifetime = 60
        
        # Enhanced color schemes
        self.color_palettes = [
            [(255, 50, 50), (50, 255, 50), (50, 50, 255)],
            [(255, 100, 0), (0, 255, 200), (148, 0, 255)],
            [(255, 200, 50), (255, 50, 150), (50, 150, 255)]
        ]
        self.current_palette = 0
        
    def process_audio(self, data):
        bass, mids, highs, is_peak, relative_bands = super().process_audio(data)
        
        # Simplified spectrum processing
        fft_data = np.abs(fft(data)[:len(data)//2])
        fft_data = np.log10(fft_data + 1)
        
        # Downsample spectrum data to prevent overflow
        target_size = 64  # Reduced size
        downsampled = np.mean(fft_data.reshape(-1, len(fft_data)//target_size), axis=1)
        self.spectrum_history.append(downsampled)
        
        return bass, mids, highs, is_peak, relative_bands, True
    
    def spawn_particles(self):
        # Spawn particles on beat
        num_particles = np.random.randint(5, 15)
        for _ in range(num_particles):
            angle = np.random.uniform(0, 2*np.pi)
            speed = np.random.uniform(2, 8)
            velocity = np.array([np.cos(angle), np.sin(angle)]) * speed
            pos = np.random.rand(2) * [self.width, self.height]
            self.particles.append({
                'pos': pos,
                'vel': velocity,
                'life': self.particle_lifetime,
                'color': self.color_palettes[self.current_palette][np.random.randint(3)]
            })
    
    def update_particles(self):
        for particle in self.particles[:]:
            particle['pos'] += particle['vel']
            particle['vel'] *= 0.98  # Drag
            particle['life'] -= 1
            if particle['life'] <= 0:
                self.particles.remove(particle)
    
    def draw_spectrum_flow(self):
        if len(self.spectrum_history) < 2:
            return
            
        self.flow_surface.fill((0, 0, 0))
        height = self.flow_surface.get_height()
        width = self.flow_surface.get_width()
        
        try:
            for i, spectrum in enumerate(self.spectrum_history):
                points = []
                step = len(spectrum) // width if len(spectrum) > width else 1
                
                for j in range(0, min(len(spectrum), width)):
                    try:
                        value = float(spectrum[j * step])
                        y_pos = int(height * (1 - value / (np.max(spectrum) + 1e-6)))
                        points.append((j, y_pos))
                    except (ValueError, IndexError):
                        continue
                        
                if points:
                    color_intensity = int(255 * (i / len(self.spectrum_history)))
                    pygame.draw.lines(self.flow_surface, 
                                    (color_intensity, color_intensity//2, color_intensity//4),
                                    False, points, 1)
        except Exception as e:
            print(f"Spectrum visualization error: {e}")
            pass

    
    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        with self.speakers.recorder(samplerate=self.SAMPLE_RATE) as mic:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            running = False
                        elif event.key == pygame.K_SPACE:
                            # Cycle through color palettes
                            self.current_palette = (self.current_palette + 1) % len(self.color_palettes)
                
                data = mic.record(numframes=self.CHUNK)
                bass, mids, highs, is_peak, relative_bands, is_beat = self.process_audio(data)
                
                self.update_nodes(bass, mids, highs, is_peak, relative_bands)
                self.update_particles()
                
                self.screen.fill((0, 0, 0))
                self.draw_nodes_and_edges(bass, mids, highs)
                
                # Draw particles
                for particle in self.particles:
                    alpha = int(255 * (particle['life'] / self.particle_lifetime))
                    color = tuple(map(lambda x: int(x * alpha/255), particle['color']))
                    pygame.draw.circle(self.screen, color, 
                                    particle['pos'].astype(int), 3)
                
                # Draw spectrum flow
                self.draw_spectrum_flow()
                self.screen.blit(self.flow_surface, (0, 0))
                
                pygame.display.flip()
                clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    visualizer = EnhancedAudioVisualizer()
    visualizer.run()