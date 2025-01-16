import numpy as np
import soundcard as sc
import pygame
from scipy.fft import fft
from scipy.spatial.distance import pdist, squareform
from collections import deque

class EnhancedAudioVisualizer:
    def __init__(self):
        pygame.init()
        self.width = 900
        self.height = 900
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Enhanced Audio Visualizer")
        
        # Audio setup with improved buffer
        self.SAMPLE_RATE = 44100
        self.CHUNK = 2048
        self.speakers = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)
        
        # Enhanced frequency analysis
        self.num_freq_bands = 8
        self.freq_band_history = deque(maxlen=60)
        self.intensity_change_history = deque(maxlen=5)
        self.smoothing_factor = 0.3
        
        # Visual effects
        self.blur_surface = pygame.Surface((self.width, self.height))
        self.blur_surface.set_alpha(20)
        self.trail_alpha = 20
        
        # Motion control parameters
        self.global_speed_factor = 0.5
        self.reaction_speed_factor = 1.4
        self.damping_factor = 0.98
        
        # Particle system setup
        self.setup_particle_system()
        
    def setup_particle_system(self):
        self.layers = []
        
        freq_edges = np.logspace(np.log10(20), np.log10(20000), self.num_freq_bands + 1)
        
        freq_ranges = []
        nyquist_freq = self.SAMPLE_RATE / 2
        for i in range(self.num_freq_bands):
            start = freq_edges[i] / nyquist_freq
            end = freq_edges[i+1] / nyquist_freq
            freq_ranges.append((start, end))
        
        base_colors = [
            (255, 80, 80),    # Red
            (255, 170, 80),   # Orange
            (255, 255, 80),   # Yellow
            (170, 255, 80),   # Light green
            (80, 255, 80),    # Green
            (100, 255, 255),  # Cyan
            (130, 200, 255),  # Light blue
            (200, 130, 255),  # Purple
        ]
        
        min_intensity = 0.3
        
        # Pre-calculate random values for better performance
        max_particles = 100  # Maximum number of particles you might use
        self.random_positions = np.random.rand(max_particles * self.num_freq_bands, 2)
        self.random_phases = np.random.rand(max_particles * self.num_freq_bands) * 2 * np.pi
        
        for i in range(self.num_freq_bands):
            freq_factor = i / (self.num_freq_bands - 1)
            num_particles = 50  # You can adjust this number
            base_size = 3.5
            
            # Use pre-calculated random values
            start_idx = i * max_particles
            end_idx = start_idx + num_particles
            
            self.layers.append({
                'num_particles': num_particles,
                'positions': self.random_positions[start_idx:end_idx] * [self.width, self.height],
                'velocities': np.zeros((num_particles, 2)),
                'accelerations': np.zeros((num_particles, 2)),
                'sizes': np.ones(num_particles) * base_size,
                'base_sizes': np.ones(num_particles) * base_size,
                'colors': np.zeros((num_particles, 3)),
                'base_color': np.array(base_colors[i]),
                'frequency_range': freq_ranges[i],
                'orbital_speed': 2.0 * self.global_speed_factor,
                'center_attraction': 0.0005,
                'chaos_factor': 0.2,
                'pulse_phase': self.random_phases[start_idx:end_idx],
                'energy_level': min_intensity,
                'reaction_strength': 1.2,
                'connection_cache': None,  # Add connection cache
                'last_update': 0  # Add timestamp for cache
            })
    
    def process_audio(self, data):
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        # Enhanced FFT processing
        fft_data = fft(data * np.hanning(len(data)))
        fft_freq = np.abs(fft_data[:len(fft_data)//2])
        fft_freq = np.log10(fft_freq + 1)
        
        # Process frequency bands
        #print("\nFrequency bands:")  # Debug print
        bands = []
        for i in range(self.num_freq_bands):
            start_idx = int(len(fft_freq) * self.layers[i]['frequency_range'][0])
            end_idx = int(len(fft_freq) * self.layers[i]['frequency_range'][1])
            band_intensity = np.mean(fft_freq[start_idx:end_idx])
            bands.append(band_intensity)
            #print(f"Band {i}: {band_intensity}")  # Debug print
        
        # Normalize and smooth
        bands = np.array(bands) / (np.max(bands) + 1e-6)
        if self.freq_band_history:
            prev_bands = self.freq_band_history[-1]
            bands = prev_bands * (1 - self.smoothing_factor) + bands * self.smoothing_factor
        
        # Calculate intensity changes
        if self.freq_band_history:
            intensity_changes = bands - self.freq_band_history[-1]
        else:
            intensity_changes = np.zeros_like(bands)
        
        self.freq_band_history.append(bands)
        self.intensity_change_history.append(intensity_changes)
        
        return bands, intensity_changes
    
    def update_particle_layer(self, layer, intensity, intensity_change):
        # Higher minimum intensity for visibility
        min_intensity = 0.005
        intensity = max(intensity, min_intensity)
        
        center = np.array([self.width/2, self.height/2])
        forbidden_radius = 30  # Keep the same boundary
        optimal_radius = 300   # New parameter for desired spread
        max_velocity = 14
        
        layer['energy_level'] = max(
            min_intensity,
            layer['energy_level'] * 0.98 + intensity * 0.005
        )
        reaction_multiplier = 5.0 + np.abs(intensity_change) * 2000.0
        layer['reaction_strength'] = min(0.05, layer['reaction_strength'] * 0.95 + reaction_multiplier * 0.5)
        layer['pulse_phase'] += 0.02 * (layer['energy_level'] + min_intensity)
        
        for i in range(layer['num_particles']):
            to_center = center - layer['positions'][i]
            dist = np.linalg.norm(to_center)
            
            if dist < 1e-6:
                continue
                
            pulse = np.sin(layer['pulse_phase'][i] + dist * 0.01) * 0.5 + 0.4
            normalized_to_center = to_center / dist
            perpendicular = np.array([-normalized_to_center[1], normalized_to_center[0]])
            
            # Modified orbital motion based on distance
            base_orbital = perpendicular * layer['orbital_speed'] * 7.0  # Added multiplier to increase tangential velocity
            if dist < forbidden_radius + 5:  # Near boundary
                # Strong outward push
                radial_force = -normalized_to_center * 10.0
                force = base_orbital + radial_force
            else:
                # Dynamic orbital speed based on distance from optimal radius
                radius_diff = (dist - optimal_radius) / optimal_radius
                orbital_mod = 1.0 - 0.4 * radius_diff  # Slower when far out
                radial_mod = 7.0 * radius_diff  # Gentle push towards optimal radius
                force = base_orbital * orbital_mod + normalized_to_center * radial_mod
            
            # Scale force by intensity
            force *= (0.5 + intensity * 0.5)
            #print("Force:", np.linalg.norm(force))  # Print magnitude of force
            
            # Add explosive force for intensity changes
            if intensity_change > 0:
                angle = np.arctan2(to_center[1], to_center[0])
                variance = np.array([np.cos(angle * 3), np.sin(angle * 3)])
                explosive_force = (normalized_to_center + variance) * intensity_change * 500.0
                force += explosive_force * self.reaction_speed_factor
            
            # Enhanced chaos based on distance
            dist_factor = np.clip((dist - forbidden_radius) / (optimal_radius - forbidden_radius), 0, 1)
            chaos_scale = 0.5 + 0.5 * dist_factor  # More chaos further out
            chaos = np.random.randn(2) * layer['chaos_factor'] * layer['energy_level'] * chaos_scale
            
            # Update motion
            total_force = force * self.global_speed_factor
            layer['accelerations'][i] = (total_force + chaos) * self.global_speed_factor
            layer['velocities'][i] = (
                layer['velocities'][i] * self.damping_factor + 
                layer['accelerations'][i] * self.reaction_speed_factor
            )
            # Add:
            velocity_magnitude = np.linalg.norm(layer['velocities'][i])
            #print("Velocity:", np.linalg.norm(layer['velocities'][i]))  # Print velocity magnitude

            if velocity_magnitude > max_velocity:
                layer['velocities'][i] = layer['velocities'][i] * (max_velocity / velocity_magnitude)
            
            # Update position
            new_position = layer['positions'][i] + layer['velocities'][i]

            # Check if new position would cross the center boundary
            to_center_new = center - new_position
            new_dist = np.linalg.norm(to_center_new)
            
            if new_dist < forbidden_radius:
                # If would cross boundary, reflect the velocity and position
                reflection = normalized_to_center * 2 * (new_dist - forbidden_radius)
                new_position += reflection
                
                # Reflect velocity with some energy loss
                reflection_normal = to_center_new / new_dist
                velocity_normal = np.dot(layer['velocities'][i], reflection_normal) * reflection_normal
                velocity_tangent = layer['velocities'][i] - velocity_normal
                layer['velocities'][i] = velocity_tangent - velocity_normal * 1.5  # 20% energy loss
            
            # After position update
            layer['positions'][i] = new_position
            layer['positions'][i] = np.nan_to_num(layer['positions'][i], nan=0.0)  # Replace NaN with 0
            
            # Boundary handling for screen edges
            for dim in range(2):
                if layer['positions'][i][dim] < 0:
                    layer['positions'][i][dim] += [self.width, self.height][dim]
                elif layer['positions'][i][dim] > [self.width, self.height][dim]:
                    layer['positions'][i][dim] -= [self.width, self.height][dim]
            
            # Update size with pulse effect
            size_factor = 1.0 + pulse * intensity * 0.5 + np.abs(intensity_change) * 4.0
            layer['sizes'][i] = np.clip(layer['base_sizes'][i] * size_factor, 1.0, 20.0)  # Clip to valid range
            layer['sizes'][i] = np.nan_to_num(layer['sizes'][i], nan=1.0)  # Replace NaN with minimum size
            
            #Enhanced color calculation
            speed = np.linalg.norm(layer['velocities'][i])
            energy_brightness = np.clip(0.4 + 0.6 * layer['energy_level'], 0, 1)  # Clip to valid range
            speed_factor = np.clip(speed / 8, 0, 1)  # Clip to valid range

            # Enhanced pulse-based color modification
            pulse = np.sin(layer['pulse_phase'][i] + dist * 0.01) * 0.5 + 0.5

            # Base color with safety check
            base = layer['base_color'] * max(energy_brightness, 0.4)
            speed_influence = layer['base_color'] * speed_factor * (intensity + min_intensity) * 0.3
            pulse_influence = layer['base_color'] * pulse * (intensity + min_intensity) * 0.2

            # Combine with safety checks
            layer['colors'][i] = np.nan_to_num(base + speed_influence + pulse_influence, nan=0.0)

            # Double ensure colors stay within valid range and handle any remaining NaN
            layer['colors'][i] = np.clip(np.nan_to_num(layer['colors'][i], nan=0.0), 0, 255)
    
    def draw_particle_connections(self, layer, intensity, intensity_change):
        min_intensity = 0.9
        base_intensity = max(intensity, min_intensity)
        max_dist = 200
        min_dist = 160
        max_edges_per_node = 4  # New parameter - adjust this value as needed
        
        current_time = pygame.time.get_ticks()
        cache_duration = 50  # Update connections every 50ms
        
        # Use cached connections if recent enough
        if (layer['connection_cache'] is not None and 
            current_time - layer['last_update'] < cache_duration):
            connections = layer['connection_cache']
        else:
            # Calculate new connections
            positions = layer['positions']
            connections = []
            
            # Use numpy operations for distance calculation
            distances = squareform(pdist(positions))
            
            # Keep track of edges per node
            edges_count = {i: 0 for i in range(len(positions))}
            
            # Find valid connections and apply max edges limit
            valid_connections = np.where((distances > min_dist) & (distances < max_dist))
            
            # Sort connections by distance to prioritize closer particles
            connection_data = [(i, j, distances[i, j]) for i, j in zip(*valid_connections) if i < j]
            connection_data.sort(key=lambda x: x[2])  # Sort by distance
            
            for i, j, dist in connection_data:
                # Only add connection if both nodes haven't reached their max edges
                if edges_count[i] < max_edges_per_node and edges_count[j] < max_edges_per_node:
                    connections.append((i, j, dist))
                    edges_count[i] += 1
                    edges_count[j] += 1
            
            # Cache the results
            layer['connection_cache'] = connections
            layer['last_update'] = current_time
        
        # Draw connections
        color_boost = 1.0
        if np.array_equal(layer['base_color'], (100, 255, 255)):
            color_boost = 4.0
        elif np.array_equal(layer['base_color'], (130, 200, 255)):
            color_boost = 5.0
        elif np.array_equal(layer['base_color'], (200, 130, 255)):
            color_boost = 5.0
        else:
            color_boost = 4.0
        
        # Draw all connections at once using pygame.draw.lines
        for i, j, dist in connections:
            connection_intensity = (1 - dist/max_dist) * (base_intensity + 0.1) * color_boost
            if intensity_change > 0:
                connection_intensity *= (1 + intensity_change * 2.5)
            
            color1 = layer['colors'][i]
            color2 = layer['colors'][j]
            avg_color = (color1 + color2) / 2
            
            connection_intensity = np.clip(connection_intensity, 0, 1)
            safe_color = tuple(map(int, np.clip(avg_color * max(connection_intensity, 0.2), 0, 255)))
            
            pygame.draw.line(
                self.screen,
                safe_color,
                layer['positions'][i].astype(int),
                layer['positions'][j].astype(int),
                max(1, int(connection_intensity * 2.5))
            )

    def draw_frequency_key(self):
        # Key positioning and sizing
        key_x = 10
        key_y = 10
        key_width = 120
        key_height = 15
        padding = 3
        text_offset = 5
        
        # Initialize pygame font
        if not hasattr(self, 'font'):
            self.font = pygame.font.SysFont('Arial', 10)
        
        # Create semi-transparent background
        key_surface = pygame.Surface((key_width + 60, (key_height + padding) * self.num_freq_bands))
        key_surface.set_alpha(150)
        key_surface.fill((0, 0, 0))
        self.screen.blit(key_surface, (key_x - padding, key_y - padding))
        
        # Get frequency ranges
        freq_edges = np.logspace(np.log10(20), np.log10(20000), self.num_freq_bands + 1)
        
        # Draw each frequency band
        for i in range(self.num_freq_bands):
            # Draw colored rectangle
            pygame.draw.rect(
                self.screen,
                self.layers[i]['base_color'],
                (key_x, key_y + i * (key_height + padding), key_width, key_height)
            )
            
            # Add frequency text
            freq_text = f"{int(freq_edges[i])}-{int(freq_edges[i+1])}Hz"
            text_surface = self.font.render(freq_text, True, (255, 255, 255))
            self.screen.blit(text_surface, (key_x + key_width + text_offset, 
                                        key_y + i * (key_height + padding)))

    
    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        # Pre-allocate surfaces for better performance
        self.particle_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        with self.speakers.recorder(samplerate=self.SAMPLE_RATE) as mic:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            running = False
                
                data = mic.record(numframes=self.CHUNK)
                frequency_bands, intensity_changes = self.process_audio(data)
                
                self.screen.blit(self.blur_surface, (0, 0))
                self.screen.fill((0, 0, 0, self.trail_alpha))
                
                # Clear particle surface
                self.particle_surface.fill((0, 0, 0, 0))
                
                # Update and draw all particles
                for layer_idx, layer in enumerate(self.layers):
                    intensity = frequency_bands[layer_idx]
                    intensity_change = intensity_changes[layer_idx]
                    
                    self.update_particle_layer(layer, intensity, intensity_change)
                    self.draw_particle_connections(layer, intensity, intensity_change)
                    
                    # Draw particles in batches
                    positions = layer['positions'].astype(int)
                    colors = [tuple(map(int, np.clip(c, 0, 255))) for c in layer['colors']]
                    sizes = np.clip(layer['sizes'], 1, None).astype(int)
                    
                    for pos, color, size in zip(positions, colors, sizes):
                        pygame.draw.circle(self.particle_surface, color, pos, size)
                
                # Blit particle surface to screen
                self.screen.blit(self.particle_surface, (0, 0))
                self.draw_frequency_key()
                
                pygame.display.flip()
                clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    visualizer = EnhancedAudioVisualizer()
    visualizer.run()