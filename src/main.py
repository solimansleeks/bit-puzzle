import time
import yaml
import logging
from src.solvers import (
    ECCSolver,
    MathematicalConstantsSolver,
    SideChannelSolver,
    DifferentialAnalysisSolver,
    QuantumInspiredSolver,
    BlockchainAnalysisSolver
)
from src.data_processing.data_processor import DataProcessor
from src.blockchain_analysis.blockchain_analyzer import BlockchainAnalyzer
from src.machine_learning.ml_models import MLModels
from src.visualization.data_visualizer import DataVisualizer
from data_processing.data_processor import DataProcessor
from blockchain_analysis.blockchain_analyzer import BlockchainAnalyzer
from machine_learning.ml_models import MLModels
from visualization.data_visualizer import DataVisualizer

def setup_logging(config):
    logging.basicConfig(
        level=config['logging']['level'],
        filename=config['logging']['file'],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    setup_logging(config)
    logger = logging.getLogger(__name__)

    target_address = config['puzzle']['address']
    public_key = config['puzzle']['public_key']
    difficulty = config['puzzle']['difficulty']

    # Initialize new components
    data_processor = DataProcessor()
    blockchain_analyzer = BlockchainAnalyzer()
    ml_models = MLModels()
    data_visualizer = DataVisualizer()

    # Load and process data
    data = data_processor.load_data(config['data']['file_path'])
    processed_data = data_processor.preprocess_data(data)

    # Perform blockchain analysis
    blockchain_data = blockchain_analyzer.get_address_info(target_address)

    # Train and evaluate ML model
    if config['ml']['enabled']:
        X = processed_data.drop(columns=[config['data']['target_column']])
        y = processed_data[config['data']['target_column']]
        model, metrics = ml_models.train_and_evaluate(X, y, config['ml']['model_name'])
        logger.info(f"ML model evaluation metrics: {metrics}")

    # Visualize data
    if config['visualization']['enabled']:
        data_visualizer.plot_histogram(processed_data[config['data']['target_column']], 
                                       'Target Distribution', 'Value', 'Frequency', 
                                       'target_distribution.png')

    solvers = [
        ECCSolver(config['solvers']['ecc_solver']),
        MathematicalConstantsSolver(config['solvers']['mathematical_constants_solver']),
        SideChannelSolver(config['solvers']['side_channel_solver']),
        DifferentialAnalysisSolver(config['solvers']['differential_analysis_solver']),
        QuantumInspiredSolver(config['solvers']['quantum_inspired_solver']),
        BlockchainAnalysisSolver(config['solvers']['blockchain_analysis_solver'])
    ]

    for solver in solvers:
        if solver.config['enabled']:
            logger.info(f"Running {solver.__class__.__name__}...")
            try:
                result = solver.solve(target_address, public_key, difficulty)
                if result:
                    logger.info(f"Solution found by {solver.__class__.__name__}: {result}")
                    return result
                else:
                    logger.info(f"{solver.__class__.__name__} failed to find a solution.")
            except Exception as e:
                logger.error(f"Error in {solver.__class__.__name__}: {str(e)}")

    logger.info("No solution found by any solver.")
    return None

if __name__ == "__main__":
    main()
