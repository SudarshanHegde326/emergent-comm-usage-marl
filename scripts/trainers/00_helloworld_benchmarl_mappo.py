import torch
from benchmarl.algorithms import MappoConfig
from benchmarl.environments import PettingzooConfig
from benchmarl.experiments import Experiment

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Targeting active processing device: {device}")

    env_config = PettingzooConfig.get_from_yaml(
        task="simple_spread",
        parallel_envs=2,          
        max_steps=25,             
    )

    algorithm_config = MappoConfig.get_from_yaml(
        share_param_critic=True,  
        lr=5e-4,                  
        critic_lr=5e-4,
    )

    experiment = Experiment(
        task=env_config,
        algorithm=algorithm_config,
        seed=0,                   
        device=device,
        total_steps=50_000,       
        loggers=["wandb", "csv"], 
    )

    print("Launching baseline training optimization run...")
    experiment.run()

if __name__ == "__main__":
    main()
