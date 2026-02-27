
import argparse, re, numpy as np, gymnasium as gym, minihack, os
from minihack import LevelGenerator, MiniHackSkill

def make_des():
    gen = LevelGenerator(w=3, h=3)
    gen.add_object("skeleton key", "(")
    gen.add_object("food ration", "%")
    gen.add_object("dagger", ")")
    gen.add_object("leather armor", "[")
    gen.add_trap("fire")
    gen.add_fountain()
    gen.add_stair_down()
    gen.add_door(state="closed")
    return gen.get_des()

class DenseItemRoom(MiniHackSkill):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("observation_keys", ("glyphs", "blstats", "message", "chars"))
        kwargs["max_episode_steps"] = 200
        super().__init__(*args, des_file=make_des(), autopickup=True, **kwargs)

gym.register(id="MiniHack-DenseRoom-v0", entry_point=DenseItemRoom)

KEEP_RE = re.compile(
    r"^([a-z] - |the door|a tower of flame|you fall|you feel|the fountain|you burn|you eat|you are burned)",
    re.IGNORECASE
)

def normalize(msg):
    msg = re.sub(r"[+-]\d+", "<bonus>", msg)
    msg = re.sub(r"\d+", "<n>", msg)
    return msg.lower().strip().rstrip(".")

def collect(args):
    env = gym.make("MiniHack-DenseRoom-v0")
    states, actions, next_states, rewards, dones, messages = [], [], [], [], [], []
    seen_messages, no_new_count, total_steps, ep = set(), 0, 0, 0
    print("World Knowledge Collection (3x3)\n")
    while True:
        obs, _ = env.reset()
        done, new_this_ep = False, 0
        while not done:
            s = obs["glyphs"].copy()
            a = env.action_space.sample()
            next_obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            msg = bytes(next_obs["message"]).decode("utf-8").rstrip("\x00").strip()
            if msg and KEEP_RE.match(msg):
                norm = normalize(msg)
                if norm not in seen_messages:
                    seen_messages.add(norm)
                    new_this_ep += 1
                    print(f"  [NEW #{len(seen_messages):3d}] {norm}")
                states.append(s); actions.append(a)
                next_states.append(next_obs["glyphs"].copy())
                rewards.append(r); dones.append(float(done)); messages.append(norm)
            obs = next_obs
            total_steps += 1
        ep += 1
        no_new_count = 0 if new_this_ep > 0 else no_new_count + 1
        if ep % 20 == 0:
            print(f"Ep {ep} | Steps {total_steps} | Unique {len(seen_messages)} | No-new {no_new_count}/{args.patience}")
        if no_new_count >= args.patience:
            print(f"Converged after {ep} episodes!"); break
        if ep >= args.max_episodes:
            break
    env.close()
    os.makedirs("data", exist_ok=True)
    np.savez_compressed("data/world_knowledge.npz",
        states=np.array(states, dtype=np.int16), actions=np.array(actions, dtype=np.int16),
        next_states=np.array(next_states, dtype=np.int16), rewards=np.array(rewards, dtype=np.float32),
        dones=np.array(dones, dtype=np.float32), messages=np.array(messages))
    print(f"\nSaved {len(states)} transitions")
    print(f"\nWORLD KNOWLEDGE ({len(seen_messages)} interactions):")
    for i, m in enumerate(sorted(seen_messages)):
        print(f"  {i+1:3d}. {m}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--max_episodes", type=int, default=5000)
    collect(parser.parse_args())
