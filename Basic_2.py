import numpy as np
import soundcard as sc
import pygame
from scipy.fft import fft
import torch
from opensimplex import OpenSimplex
import colorsys
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter
import warnings
from soundcard.mediafoundation import SoundcardRuntimeWarning

warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)

class EnhancedTensorVisualizer:
    def __init__(self):
        pygame.init()
        self.width = 1200
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Enhanced Tensor Flow Audio Visualizer")
        
        # Audio setup with improved buffer
        self.SAMPLE_RATE = 44100
        self.CHUNK = 2048
        self.speakers = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)
        
        # Modified tensor field parameters for better visibility
        self.field_resolution = 40  # Reduced from 60 for less density
        self.points = torch.zeros((self.field_resolution, self.field_resolution, 2))
        self.velocities = torch.zeros((self.field_resolution, self.field_resolution, 2))
        self.field_speed = 1.0  # Reduced for smoother movement
        self.field_strength = 1.0  # Reduced for less chaos
        
        # Enhanced oscillator parameters
        self.num_oscillators = 8  # Reduced for clarity
        self.oscillators = []
        #self.setup_oscillators()
        
        
        # Improved Julia set parameters
        self.julia_size = 400
        self.julia_surface = pygame.Surface((self.julia_size, self.julia_size))
        self.julia_c = complex(-0.4, 0.6)
        self.julia_zoom = 1.2
        self.max_iter = 200  # Increased for better detail
        
        # Color and visual parameters
        self.color_offset = 0
        self.background_color = (10, 10, 15)  # Dark blue-black background
        self.line_alpha = 160  # Reduced opacity for better layering
        
        # Initialize noise generator
        self.noise_gen = OpenSimplex(seed=np.random.randint(1000))
        self.time = 0
        # Smooth transitions
        self.smoothing_factor = 0.15
        self.previous_energies = None
        
        self.setup_field()

    def setup_field(self):
        """Initialize tensor field with improved spacing"""
        x = torch.linspace(0, self.width, self.field_resolution)
        y = torch.linspace(0, self.height, self.field_resolution)
        self.grid_x, self.grid_y = torch.meshgrid(x, y, indexing='ij')
        self.points = torch.stack([self.grid_x, self.grid_y], dim=-1)
        
        # Add spacing buffer from edges
        edge_buffer = 50
        self.points = self.points * (1 - 2*edge_buffer/self.width) + edge_buffer

    def setup_oscillators(self):
        """Initialize oscillators with better spacing and varied frequencies"""
        center = np.array([self.width/2, self.height/2])
        for i in range(self.num_oscillators):
            angle = 2 * np.pi * i / self.num_oscillators
            self.oscillators.append({
                'freq': 1.5 + i * 0.5,  # More varied frequencies
                'phase': angle,
                'amplitude': 120 + i * 10,  # Varied amplitudes
                'center': center + np.array([np.cos(angle), np.sin(angle)]) * 50,
                'trail': [],
                'color': None  # Will be set dynamically
            })

    def process_audio(self, data):
        """Enhanced audio processing with focus on guitar frequencies"""
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        window = np.hanning(len(data))
        fft_data = fft(data * window)
        fft_freq = np.abs(fft_data[:len(fft_data)//2])
        fft_freq = np.log10(fft_freq + 1)
        
        # Focus on guitar-relevant frequencies
        freq_axis = np.fft.fftfreq(len(data), 1/self.SAMPLE_RATE)[:len(data)//2]
        
        # Enhanced frequency bands for guitar
        chug_band = (80, 200)  # Frequency range for typical guitar chugs
        bass = np.mean(fft_freq[np.where((freq_axis >= chug_band[0]) & (freq_axis <= chug_band[1]))])
        mids = np.mean(fft_freq[int(len(fft_freq)*0.1):int(len(fft_freq)*0.3)])
        highs = np.mean(fft_freq[int(len(fft_freq)*0.3):])
        
        # Detect sudden changes in chug band (characteristic of chugs)
        if self.previous_energies is not None:
            chug_change = bass - self.previous_energies[0]
            bass = bass * (1 + max(0, chug_change) * 5)  # Amplify sudden increases
        
        energies = np.array([bass, mids, highs])
        if self.previous_energies is None:
            self.previous_energies = energies
        else:
            energies = self.previous_energies * (1 - self.smoothing_factor) + energies * self.smoothing_factor
            self.previous_energies = energies
        
        return energies[0], energies[1], energies[2]

    def update_tensor_field(self, bass, mids, highs):
        """Update tensor field with enhanced reactivity to bass/chugs"""
        field = torch.zeros_like(self.points)
        time_scale = 0.001
        space_scale = 0.03
        
        # Amplify field strength based on bass
        base_strength = self.field_strength * (1 + bass * 2.5)  # Increased bass influence
        
        for i in range(self.field_resolution):
            for j in range(self.field_resolution):
                x = self.points[i, j, 0].item()
                y = self.points[i, j, 1].item()
                
                # Base flow pattern
                noise_val = self.noise_gen.noise3(
                    x * space_scale,
                    y * space_scale,
                    self.time * time_scale
                )
                
                # Center influence for organized patterns
                center_x, center_y = self.width/2, self.height/2
                dx, dy = x - center_x, y - center_y
                distance = np.sqrt(dx*dx + dy*dy)
                angle = np.arctan2(dy, dx)
                
                # Enhanced reactivity to bass
                flow_angle = noise_val * 4 * np.pi + angle * (1 + bass * 2)
                strength = base_strength * (1 + bass * 2)  # More bass influence
                
                # Add pulse effect on strong bass hits
                if bass > 0.5:  # Threshold for strong bass
                    pulse = np.sin(distance * 0.05 - self.time * 0.1) * bass
                    strength *= (1 + pulse)
                
                field[i, j, 0] = np.cos(flow_angle) * strength
                field[i, j, 1] = np.sin(flow_angle) * strength
        
        # More aggressive velocity updates on strong bass
        blend_factor = 0.9 - bass * 0.4  # More instant response to bass
        self.velocities = self.velocities * blend_factor + field * (1 - blend_factor)
        
        # Add impulse on bass hits
        if bass > 0.5:
            self.velocities += field * bass * 0.4
        
        self.points = self.points + self.velocities * self.field_speed

    def calculate_julia(self, bass, mids):
        """Generate high-quality Julia set"""
        julia = np.zeros((self.julia_size, self.julia_size))
        
        # Dynamic Julia parameters based on audio
        angle = self.time * 0.05 * (1 + mids * 0.9)
        radius = 0.7885 * (1 + bass * 0.1)
        c = complex(radius * np.cos(angle), radius * np.sin(angle))
        
        x = np.linspace(-1.5, 1.5, self.julia_size) / self.julia_zoom
        y = np.linspace(-1.5, 1.5, self.julia_size) / self.julia_zoom
        X, Y = np.meshgrid(x, y)
        Z = X + Y*1j
        
        # Compute Julia set with increased precision
        for i in range(self.max_iter):
            mask = np.abs(Z) < 2
            julia[mask] += 1
            Z[mask] = Z[mask]**2 + c
        
        # Enhanced coloring
        julia = julia / self.max_iter
        julia = np.power(julia, 0.5)  # Gamma correction
        
        return julia

    def draw(self, bass, mids, highs):
        """Improved drawing routine with better layering"""
        # Apply fade effect
        self.screen.fill(self.background_color)
        
        # Draw tensor field with improved visibility
        points_np = self.points.numpy()
        velocities_np = self.velocities.numpy()
        
        for i in range(0, self.field_resolution, 1):
            for j in range(0, self.field_resolution, 1):
                pos = points_np[i, j]
                vel = velocities_np[i, j]
                
                # Calculate color based on velocity and position
                vel_magnitude = np.linalg.norm(vel)
                hue = (np.arctan2(vel[1], vel[0]) / (2*np.pi) + 0.5 + self.color_offset) % 1.0
                saturation = min(1.0, 0.4 + vel_magnitude * 0.3)
                value = min(1.0, 0.6 + vel_magnitude * 0.2)
                
                rgb = tuple(int(x * 255) for x in colorsys.hsv_to_rgb(hue, saturation, value))
                
                # Draw line with proper alpha
                start_pos = tuple(map(int, pos))
                end_pos = tuple(map(int, pos + vel * 0.3))
                
                if (0 <= start_pos[0] < self.width and 
                    0 <= start_pos[1] < self.height and
                    0 <= end_pos[0] < self.width and 
                    0 <= end_pos[1] < self.height):
                    pygame.draw.line(self.screen, rgb, start_pos, end_pos, 2)
        
        
        # Draw Julia set with enhanced clarity
        julia = self.calculate_julia(bass, mids)
        julia_colored = np.zeros((julia.shape[0], julia.shape[1], 3))
        
        for i in range(3):
            hue = (self.color_offset + i/3) % 1.0
            julia_colored[:, :, i] = julia * 255
        
        julia_surface = pygame.surfarray.make_surface(julia_colored.astype(np.uint8))
        self.screen.blit(julia_surface, (self.width - self.julia_size - 20, 20))
        
        # Update color offset
        self.color_offset += 0.001
        self.time += 1

    def run(self):
        """Main loop with improved timing"""
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
                bass, mids, highs = self.process_audio(data)
                
                self.update_tensor_field(bass, mids, highs)
                self.draw(bass, mids, highs)
                
                pygame.display.flip()
                clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    visualizer = EnhancedTensorVisualizer()
    visualizer.run()