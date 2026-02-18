import wandb
import os


def init_wandb(project_name="audio-spoofing-detection", run_name=None, config=None):
    """Initialize Weights & Biases for experiment tracking"""
    
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        print("Warning: WANDB_API_KEY not found. Running without W&B logging.")
        return None
    
    default_config = {
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 16,
        "sample_rate": 16000,
        "target_duration": 6.0,
        "frame_duration": 0.020,
        "warmup_steps": 1600,
        "model": "BoundaryDetectionModel",
        "feature_extractor": "Wav2Vec2",
        "optimizer": "Adam"
    }
    
    if config:
        default_config.update(config)
    
    wandb.init(
        project=project_name,
        name=run_name,
        config=default_config
    )
    
    return wandb


def log_metrics(metrics_dict, step=None):
    """Log metrics to W&B"""
    if wandb.run is not None:
        wandb.log(metrics_dict, step=step)


def log_model_checkpoint(checkpoint_path, epoch):
    """Log model checkpoint as artifact"""
    if wandb.run is not None:
        artifact = wandb.Artifact(f"model-checkpoint-epoch-{epoch}", type="model")
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)


def finish_wandb():
    """Finish W&B run"""
    if wandb.run is not None:
        wandb.finish()
