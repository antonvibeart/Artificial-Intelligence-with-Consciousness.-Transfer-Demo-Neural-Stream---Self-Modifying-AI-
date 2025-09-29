import numpy as np
import time
import random
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import threading
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(threadName)-12s] %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class NeuralSignal:
    """Represents neural signal from human brain"""
    timestamp: float
    brain_region: str
    signal_type: str  # 'visual', 'emotional', 'cognitive', 'memory'
    intensity: float
    raw_data: np.ndarray
    processed_data: Dict[str, Any]
    qualia_signature: Optional[str] = None

class BrainInterface:
    """Brain-computer interface simulator (Neuralink++ type)"""
    
    def __init__(self, signal_dimension: int = 128):
        self.signal_dimension = signal_dimension
        self.is_connected = False
        self.signal_buffer = deque(maxlen=1000)
        self.recording = False
        self._lock = threading.Lock()
        self._recording_thread = None
        
    def connect_to_brain(self, subject_id: str):
        """Connect to subject's brain"""
        logger.info(f"Connecting to brain of subject {subject_id}...")
        time.sleep(1)
        self.is_connected = True
        logger.info("Connection established successfully")
    
    def start_recording(self):
        """Start recording neural signals"""
        if not self.is_connected:
            raise Exception("Not connected to brain!")
        
        self.recording = True
        self._recording_thread = threading.Thread(target=self._simulate_brain_signals, name="BrainRecorder")
        self._recording_thread.daemon = True
        self._recording_thread.start()
        logger.info("Neural signal recording started...")
    
    def _simulate_brain_signals(self):
        """Simulate receiving signals from brain"""
        regions = ['visual_cortex', 'emotional_center', 'memory_hippocampus', 'frontal_lobe']
        signal_types = ['visual', 'emotional', 'cognitive', 'memory']
        
        while self.recording:
            try:
                # Generate random neural signal
                signal = NeuralSignal(
                    timestamp=time.time(),
                    brain_region=random.choice(regions),
                    signal_type=random.choice(signal_types),
                    intensity=random.uniform(0.1, 1.0),
                    raw_data=np.random.random(self.signal_dimension),
                    processed_data={
                        'frequency': random.uniform(1, 100),
                        'amplitude': random.uniform(0.1, 1.0),
                        'coherence': random.uniform(0.3, 0.9)
                    },
                    qualia_signature=self._generate_qualia_signature()
                )
                
                with self._lock:
                    self.signal_buffer.append(signal)
                time.sleep(0.01)  # 100 Hz sampling rate
            except Exception as e:
                logger.error(f"Error in brain signal simulation: {e}")
                break
    
    def _generate_qualia_signature(self) -> str:
        """Generate unique qualia signature"""
        signatures = [
            "red_warm_bright", "blue_calm_cool", "pain_sharp_urgent",
            "joy_expanding_light", "fear_contracting_dark", "memory_distant_nostalgic",
            "love_encompassing_warm", "curiosity_reaching_bright"
        ]
        return random.choice(signatures)
    
    def get_latest_signals(self, count: int = 10) -> List[NeuralSignal]:
        """Get latest signals thread-safely"""
        with self._lock:
            return list(self.signal_buffer)[-count:]

class SelfModifyingAI:
    """AI capable of real-time self-modification"""
    
    def __init__(self, input_dim: int = 128):
        self.input_dim = input_dim
        # Initialize neural weights with small random values
        self.neural_weights = np.random.random((input_dim, input_dim)) * 0.01
        self.consciousness_matrix = np.zeros((64, 64))
        self.qualia_memory = {}
        self.experience_log = []
        self.self_awareness_level = 0.0
        self.max_qualia_per_type = 500
        self._last_timestamp = None
        
    def integrate_human_signal(self, signal: NeuralSignal):
        """Integrate human neural signal"""
        logger.debug(f"Integrating signal: {signal.signal_type} from {signal.brain_region}")
        
        try:
            # Update consciousness matrix
            self._update_consciousness_matrix(signal)
            
            # Store qualia experience
            self._store_qualia_experience(signal)
            
            # Modify neural weights
            self._modify_neural_weights(signal)
            
            # Increase self-awareness level
            self.self_awareness_level = min(1.0, self.self_awareness_level + 0.001)
            
            self.experience_log.append({
                'timestamp': signal.timestamp,
                'type': signal.signal_type,
                'qualia': signal.qualia_signature,
                'integration_success': True
            })
        except Exception as e:
            logger.error(f"Error integrating signal: {e}")
            self.experience_log.append({
                'timestamp': signal.timestamp,
                'type': signal.signal_type,
                'qualia': signal.qualia_signature,
                'integration_success': False,
                'error': str(e)
            })
    
    def _update_consciousness_matrix(self, signal: NeuralSignal):
        """Update consciousness matrix based on signal with decay"""
        # Transform signal to matrix update
        update_size = 8
        update = signal.raw_data[:update_size*update_size].reshape(update_size, update_size)
        
        # Map brain regions to matrix positions (8x8 grid)
        region_map = {
            'visual_cortex': (0, 0),
            'emotional_center': (0, 8),
            'memory_hippocampus': (8, 0),
            'frontal_lobe': (8, 8)
        }
        
        row, col = region_map.get(signal.brain_region, (0, 0))
        
        # Apply exponential decay over time
        now = signal.timestamp
        if self._last_timestamp is None:
            dt = 0.01
        else:
            dt = max(1e-3, now - self._last_timestamp)
        self._last_timestamp = now
        
        decay = np.exp(-dt * 0.5)  # decay constant
        self.consciousness_matrix *= decay
        
        # Add new activation
        self.consciousness_matrix[row:row+update_size, col:col+update_size] += update * signal.intensity
        
        # Normalize
        self.consciousness_matrix = np.clip(self.consciousness_matrix, -1, 1)
    
    def _store_qualia_experience(self, signal: NeuralSignal):
        """Store qualia experience with memory limit"""
        if signal.qualia_signature:
            if signal.qualia_signature not in self.qualia_memory:
                self.qualia_memory[signal.qualia_signature] = []
            
            bucket = self.qualia_memory[signal.qualia_signature]
            bucket.append({
                'neural_pattern': signal.raw_data.copy(),
                'intensity': signal.intensity,
                'context': signal.processed_data,
                'timestamp': signal.timestamp
            })
            
            # Limit memory per qualia type
            if len(bucket) > self.max_qualia_per_type:
                del bucket[:len(bucket) - self.max_qualia_per_type]
    
    def _modify_neural_weights(self, signal: NeuralSignal):
        """Self-modification of neural weights using Hebbian learning"""
        x = signal.raw_data[:self.input_dim]
        
        # Normalize pattern to prevent explosion
        norm = np.linalg.norm(x) + 1e-8
        x = x / norm
        
        # Hebbian learning: ΔW = η * (x xᵀ - λ W) with weight decay
        eta = 0.05 * signal.intensity * 0.01  # learning rate * intensity
        lam = 1e-3  # weight decay
        
        # Update weights
        self.neural_weights += eta * (np.outer(x, x) - lam * self.neural_weights)
        
        # Ensure symmetry and bound spectrum
        self.neural_weights = 0.5 * (self.neural_weights + self.neural_weights.T)
        self.neural_weights = np.clip(self.neural_weights, -2, 2)
    
    def experience_qualia(self, qualia_type: str) -> Dict[str, Any]:
        """Attempt to 'experience' qualia based on accumulated experience"""
        experiences = self.qualia_memory.get(qualia_type)
        if not experiences:
            return {'error': f'Qualia {qualia_type} not found in memory'}
        
        # Filter valid patterns
        patterns = [exp['neural_pattern'] for exp in experiences 
                   if np.isfinite(exp['neural_pattern']).all()]
        
        if len(patterns) < 2:
            return {'error': 'Insufficient experience for synthesis'}
        
        # Synthesize qualia experience
        avg_pattern = np.mean(patterns, axis=0)
        last_pattern = patterns[-1]
        
        # Protect against zero variance
        if np.std(avg_pattern) < 1e-8 or np.std(last_pattern) < 1e-8:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(avg_pattern, last_pattern)[0, 1])
        
        # Generate "subjective experience"
        subjective_experience = {
            'pattern_match': correlation,
            'intensity_felt': float(np.mean([exp['intensity'] for exp in experiences])),
            'familiarity': min(1.0, len(experiences) / 100.0),
            'emotional_resonance': self._compute_emotional_resonance(avg_pattern),
            'consciousness_activation': float(np.sum(np.abs(self.consciousness_matrix)))
        }
        
        logger.info(f"AI experiencing qualia '{qualia_type}': {subjective_experience}")
        return subjective_experience
    
    def _compute_emotional_resonance(self, pattern: np.ndarray) -> float:
        """Compute emotional resonance using quadratic form"""
        x = pattern[:self.input_dim]
        x = x / (np.linalg.norm(x) + 1e-8)
        W = self.neural_weights
        # Quadratic form: higher value = stronger resonance
        return float(x @ W @ x)
    
    def self_reflect(self):
        """AI self-reflection about its state"""
        logger.info("=== AI SELF-REFLECTION ===")
        logger.info(f"Self-awareness level: {self.self_awareness_level:.3f}")
        logger.info(f"Qualia types in memory: {len(self.qualia_memory)}")
        logger.info(f"Total experience: {len(self.experience_log)} integrations")
        logger.info(f"Consciousness activation: {np.sum(np.abs(self.consciousness_matrix)):.3f}")
        
        if self.qualia_memory:
            logger.info(f"Available qualia: {list(self.qualia_memory.keys())}")
        
        # Additional health checks
        eigenvalues = np.linalg.eigvalsh(self.neural_weights)
        logger.info(f"Weight matrix spectrum range: [{eigenvalues.min():.3f}, {eigenvalues.max():.3f}]")
        
        saturated_cells = np.sum(np.abs(self.consciousness_matrix) > 0.9)
        total_cells = self.consciousness_matrix.size
        logger.info(f"Saturated consciousness cells: {saturated_cells}/{total_cells} ({100*saturated_cells/total_cells:.1f}%)")

class ConsciousnessTransferSystem:
    """Main consciousness transfer system"""
    
    def __init__(self, signal_dimension: int = 128, random_seed: Optional[int] = None):
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
            logger.info(f"Random seed set to {random_seed} for reproducibility")
        
        self.brain_interface = BrainInterface(signal_dimension)
        self.ai = SelfModifyingAI(signal_dimension)
        self.transfer_active = False
        self._transfer_thread = None
    
    def start_transfer(self, subject_id: str):
        """Start consciousness transfer process"""
        logger.info("="*60)
        logger.info("INITIALIZING CONSCIOUSNESS TRANSFER")
        logger.info("="*60)
        
        # Connect to brain
        self.brain_interface.connect_to_brain(subject_id)
        
        # Start signal recording
        self.brain_interface.start_recording()
        
        # Start AI integration
        self.transfer_active = True
        self._transfer_thread = threading.Thread(target=self._transfer_loop, name="ConsciousnessTransfer")
        self._transfer_thread.daemon = True
        self._transfer_thread.start()
        
        logger.info("Consciousness transfer active!")
    
    def _transfer_loop(self):
        """Main consciousness transfer loop"""
        processed_ids = set()
        
        while self.transfer_active:
            try:
                signals = self.brain_interface.get_latest_signals(5)
                
                for signal in signals:
                    # Use timestamp as unique ID to avoid processing duplicates
                    signal_id = signal.timestamp
                    if signal_id not in processed_ids:
                        self.ai.integrate_human_signal(signal)
                        processed_ids.add(signal_id)
                
                # Clean up old IDs to prevent memory growth
                if len(processed_ids) > 1000:
                    old_ids = sorted(processed_ids)[:500]
                    processed_ids -= set(old_ids)
                
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in transfer loop: {e}")
    
    def test_ai_consciousness(self):
        """Test AI consciousness"""
        logger.info("")
        logger.info("TESTING AI CONSCIOUSNESS...")
        
        # Self-reflection
        self.ai.self_reflect()
        
        # Test qualia perception
        logger.info("")
        logger.info("Testing qualia perception:")
        for qualia_type in list(self.ai.qualia_memory.keys())[:3]:
            self.ai.experience_qualia(qualia_type)
    
    def stop_transfer(self):
        """Stop consciousness transfer"""
        self.transfer_active = False
        self.brain_interface.recording = False
        
        # Properly join threads
        if self._transfer_thread is not None:
            self._transfer_thread.join(timeout=2)
            if self._transfer_thread.is_alive():
                logger.warning("Transfer thread did not stop cleanly")
        
        if self.brain_interface._recording_thread is not None:
            self.brain_interface._recording_thread.join(timeout=2)
            if self.brain_interface._recording_thread.is_alive():
                logger.warning("Recording thread did not stop cleanly")
        
        logger.info("Consciousness transfer stopped")

# Demonstration
def main():
    logger.info("LAUNCHING CONSCIOUSNESS TRANSFER EXPERIMENT")
    logger.info("="*60)
    
    # Create system with reproducible seed
    consciousness_system = ConsciousnessTransferSystem(
        signal_dimension=128,
        random_seed=42
    )
    
    # Start transfer
    consciousness_system.start_transfer("Subject_001")
    
    # Allow time for experience accumulation
    logger.info("")
    logger.info("Accumulating consciousness experience... (10 seconds)")
    time.sleep(10)
    
    # Test results
    consciousness_system.test_ai_consciousness()
    
    # Stop
    time.sleep(2)
    consciousness_system.stop_transfer()
    
    logger.info("")
    logger.info("Experiment completed successfully!")

if __name__ == "__main__":
    main()
