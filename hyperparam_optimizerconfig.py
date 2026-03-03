"""
Configuration module for hyperparameter optimization system.
Centralizes all configuration parameters and environment variables.
"""
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Enumeration of available optimization methods"""
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    HYBRID = "hybrid"
    RANDOM = "random"


@dataclass
class FirebaseConfig:
    """Firebase configuration dataclass"""
    project_id: str = os.getenv("FIREBASE_PROJECT_ID", "")
    credentials_path: str = os.getenv("FIREBASE_CREDENTIALS_PATH", "")
    collection_name: str = "hyperparam_optimizations"
    batch_size: int = 100
    timeout_seconds: int = 30


@dataclass
class BayesianConfig:
    """Bayesian optimization configuration"""
    n_initial_points: int = 10
    n_iterations: int = 50
    acq_func: str = "ei"  # Expected Improvement
    random_state: int = 42
    kappa: float = 1.96  # Exploration parameter
    xi: float = 0.01  # Exploitation parameter


@dataclass
class GeneticConfig:
    """Genetic algorithm configuration"""
    population_size: int = 50
    generations: int = 100
    crossover_prob: float = 0.8
    mutation_prob: float = 0.2
    tournament_size: int = 3
    elite_size: int = 2
    gene_mutation_rate: float = 0.1


@dataclass
class OptimizationConfig:
    """Main optimization configuration"""
    method: OptimizationMethod = OptimizationMethod.HYBRID
    max_evaluations: int = 200
    timeout_minutes: int = 60
    validation_folds: int = 3
    min_score_threshold: float = 0.0
    early_stopping_patience: int = 10
    parallel_evaluations: int = os.cpu_count() or 2
    
    # Performance metrics to optimize
    primary_metric: str = "sharpe_ratio"
    secondary_metrics: List[str] = None
    
    def __post_init__(self):
        if self.secondary_metrics is None:
            self.secondary_metrics = ["max_drawdown", "win_rate", "profit_factor"]
        
        # Validate configuration
        if self.max_evaluations <= 0:
            raise ValueError("max_evaluations must be positive")
        if self.parallel_evaluations <= 0:
            raise ValueError("parallel_evaluations must be positive")
        
        logger.info(f"Optimization config initialized with method: {self.method.value}")


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    file_path: Optional[str] = "logs/optimization.log"
    max_file_size_mb: int = 10
    backup_count: int = 5
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self):
        self.firebase = FirebaseConfig()
        self.bayesian = BayesianConfig()
        self.genetic = GeneticConfig()
        self.optimization = OptimizationConfig()
        self.logging = LoggingConfig()
        
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate all configuration parameters"""
        try:
            # Check Firebase configuration if enabled
            if os.getenv("USE_FIREBASE", "false").lower() == "true":
                if not self.firebase.project_id:
                    logger.warning("Firebase project_id not set")
                if not self.firebase.credentials_path:
                    logger.warning("Firebase credentials_path not set")
            
            # Validate optimization parameters
            if self.optimization.parallel_evaluations > 8:
                logger.warning("High parallel_evaluations may cause resource issues")
            
            logger.info("Configuration validation completed successfully")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization"""
        return {
            "firebase": self.firebase.__dict__,
            "bayesian": self.bayesian.__dict__,
            "genetic": self.genetic.__dict__,
            "optimization": self.optimization.__dict__,
            "logging": self.logging.__dict__
        }


# Global configuration instance
config = ConfigManager()