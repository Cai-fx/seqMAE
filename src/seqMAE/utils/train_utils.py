from typing import Callable
from flax.core import frozen_dict
from flax.training import train_state
import optax
import jax
import jax.numpy as jnp



class trainState(train_state.TrainState):
    batch_stats: dict 


def zero_grads():
    def init_fn(_): 
        return ()
    def update_fn(updates, state, params=None):
        return jax.tree_map(jnp.zeros_like, updates), ()
    return optax.GradientTransformation(init_fn, update_fn)


def process_nested_dict(nested_dict : frozen_dict.FrozenDict, key : str):
    def recursive_process(nested_dict, cur_val=True):
        processed_dict = {}
        for k, v in nested_dict.items():
            # print(k, cur_val)
            if isinstance(v, dict) or isinstance(v, frozen_dict.FrozenDict):
                processed_dict[k] = recursive_process(v, cur_val=(k != key) and cur_val)
            else:
                processed_dict[k] = (k!= key) and cur_val  
        return processed_dict

    return frozen_dict.freeze(recursive_process(nested_dict))

