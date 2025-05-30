import flax.linen as nn 
import optax 
import jax 
import jax.numpy as jnp 
import numpy as np 
from tqdm import tqdm


class logistic_reg(nn.Module):
    layer_features = (1,)
    activation_func = nn.relu 
    @nn.compact 
    def __call__(self, x):
        '''
        x: (n_sample, n_metrics), n_metrics=4
        '''
        for i, l in enumerate(self.layer_features):
            x = nn.Dense(features=l, 
                         use_bias=True)(x)
            if i!=len(self.layer_features)-1:
                x = self.activation_func(x)
        return x 

def compute_metrics(y_pred, y_true):
    return optax.sigmoid_binary_cross_entropy(logits=y_pred, labels=y_true).mean()

@jax.jit 
def train_step(state, batch):
    def loss(params):
        y = state.apply_fn({"params":params},
                        batch['x'])
        l = optax.sigmoid_binary_cross_entropy(logits=y, labels=batch['y']).mean()
        return l 
    
    grads_func = jax.grad(loss)
    grads = grads_func(state.params)
    new_state = state.apply_gradients(grads=grads)
    y_new = new_state.apply_fn({"params":new_state.params}, batch['x'])
    # y_new_val = new_state.apply_fn({"params":new_state.params}, batch['val_x'])
    
    metrics = {'l': compute_metrics(y_new, batch['y']), 
            #    'val_l': compute_metrics(y_new_val, batch['val_y'])
               }
    return new_state, metrics 

def train_epoch(ds, state, batch_size=256, rng=jax.random.PRNGKey(987)):
    '''
    ds['x'] : (n_peak, 4)
    ds['y'] : (n_peak, )
    '''
    ## create batch 
    n_batch = ds['x'].shape[0]//batch_size
    batch_id = jax.random.permutation(key=rng, x=jnp.arange(n_batch*batch_size))
    batch_id = jnp.reshape(batch_id, (n_batch, batch_size))
    
    batch_metrics_np = []
    for i, bid in enumerate(batch_id):
        batch = {'x': ds['x'][bid], 'y': ds['y'][bid],
                 'val_x': ds['val_x'], 'val_y': ds['val_y']}
        state, metric = train_step(state, batch)
        batch_metrics_np.append(metric)

    metrics = {k: np.mean([metrics[k] for metrics in batch_metrics_np])
                for k in batch_metrics_np[0]}
    
    val_y = state.apply_fn({"params":state.params}, batch['val_x'])
    metrics['val_l'] = compute_metrics(val_y, batch['val_y'])
    return state, metrics 

def train_epochs(ds,
                 state,
                 n_epoch = 20,
                 batch_size=256,
                 prng = jax.random.PRNGKey(222),
                 ):
    epoch_metrics = []
    for epoch in tqdm(np.arange(n_epoch)):
        rng, prng = jax.random.split(prng)
        state, metrics = train_epoch(ds, state=state, batch_size=batch_size, rng=rng)
        epoch_metrics.append(metrics)
        # print(f"{epoch}: l={metrics['l']}, val_l={metrics['val_l']}")
    
    epoch_metrics_tidy = {key:np.array([epoch_metrics[i][key] for i in range(len(epoch_metrics))]) 
                            for key in epoch_metrics[0].keys()}
    return state, epoch_metrics_tidy


def split_train_val_test(n_sample:int, rng=jax.random.PRNGKey(0)):
    r1, r2 = jax.random.split(rng)
    n_train = int(n_sample*0.7)
    n_val = int(n_sample*0.1)
    # n_test = n_sample - n_train - n_val 
    
    train_id = jax.random.choice(r1, n_sample, shape=(n_train, ), replace=False).sort()
    rest_id = jnp.setdiff1d(np.arange(n_sample), train_id)
    val_id = jax.random.choice(r2, rest_id, shape=(n_val, ), replace=False).sort()
    test_id = jnp.setdiff1d(rest_id, val_id).sort()
    return train_id, test_id, val_id