import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from unitree_env_paper import UnitreeEnv
from torch.nn import ELU
from stable_baselines3.common.vec_env import VecNormalize # <--- Import this
from stable_baselines3.common.callbacks import CheckpointCallback

class CheckpointCallbackWithStats(CheckpointCallback):
    """
    Callback for saving a model AND its VecNormalize statistics every ``save_freq`` steps.
    """
    def _on_step(self) -> bool:
        # 1. Call the parent method to save the model (.zip)
        super()._on_step()

        # 2. Check if the parent actually saved (using the same frequency check)
        if self.n_calls % self.save_freq == 0:
            # Construct a matching filename for the stats
            # e.g., models/Backup/rl_model_v2_100000_steps_vecnormalize.pkl
            stats_path = os.path.join(
                self.save_path, 
                f"{self.name_prefix}_{self.num_timesteps}_steps_vecnormalize.pkl"
            )

            # Save the normalization stats
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(stats_path)
                if self.verbose > 1:
                    print(f"Saved VecNormalize stats to {stats_path}")
            else:
                print("Warning: VecNormalize not found, only model saved.")
        
        return True



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Unitree Go2 robot.")
    parser.add_argument('--gui', action='store_true', help="Enable GUI rendering.")
    args = parser.parse_args()

    TENSORBOARD_LOG_DIR = "./ppo_go2_tensorboard/"
    num_cpu = 16
    model_path = '../../../unitree_mujoco/unitree_robots/go2/scene_ground.xml'

    vec_env_path = "models/Current/vec_normalize_Gait.pkl"

    vec_env = make_vec_env(
        UnitreeEnv,
        n_envs=num_cpu,
        seed=0,
        #vec_env_cls=SubprocVecEnv,
        env_kwargs={"model_path": model_path, "frame_skip": 10}
    )
    if os.path.exists(vec_env_path):
        print(f"Loading VecNormalize statistics from {vec_env_path}")
        vec_env = VecNormalize.load(vec_env_path, vec_env)
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model_name = "Train_Paper_Gait_Directions_More_batches"
    folder = "models/Current/"
    model_save_path = folder + model_name + ".zip"
    checkpoint_dir = "models/Backup/" + model_name + "/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    


    if os.path.exists(model_save_path):
        print(f"--- Loading model and continuing training on {num_cpu} environments ---")
        model = PPO.load(model_save_path, env=vec_env, tensorboard_log=TENSORBOARD_LOG_DIR)
    else:
        policy_kwargs = dict(
            activation_fn=ELU,
            net_arch=dict(pi=[512, 256, 128], qf=[512, 256, 128]),
            log_std_init=-0.7
        )
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            device="cuda",  # <-- Use GPU if available
            n_steps=4096, 
            
            # 2. Increase Batch Size for stable gradients
            batch_size=1024, # Larger batches reduce noise in gradients
            
            # 3. Lower Learning Rate slightly for fine-tuning mixed gaits
            learning_rate=1e-4, 
            
            # 4. More epochs to squeeze juice out of data
            n_epochs=10,
            ent_coef=0.005,
            clip_range=0.2,
            gae_lambda=0.95,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log=TENSORBOARD_LOG_DIR,
        )

    checkpoint_callback = CheckpointCallbackWithStats(
        save_freq=100_000 // num_cpu,
        save_path=checkpoint_dir,
        name_prefix="rl_model_v2_"
    )

    try:
        print("Starting training...")
        model.learn(
            total_timesteps=100_000_000_000,
            reset_num_timesteps=False,
            callback=checkpoint_callback
        )
    except KeyboardInterrupt:
        print("\n--- Training interrupted by user ---")
    finally:
        print(f"Saving final model to {model_save_path}")
        model.save(model_save_path.replace(".zip", ""))
        print("Model saved. Exiting.")
        
        vec_env.save("models/Current/vec_normalize.pkl") # <--- Save this
        vec_env.close()
