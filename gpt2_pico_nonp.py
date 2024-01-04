import numpy as np
from math import tanh, sqrt, pi

def gelu(x):
    """
    Applies Gaussian Error Linear Units activation function to the input array

    Args:
        x: Input array.
    
    Returns:
        An array with applied activation function.
    """
    def gelu_single_value(val: float) -> float:
        return 0.5 * val * (1 + math.tanh(math.sqrt(2 / math.pi) * (val + 0.044715 * val**3)))

    if isinstance(x, list):
        return [gelu_single_value(val) for val in x]
    else:
        gelu_single_value(x)

def softmax(x):
    """
    Applies softmax function to the input array.

    Args:
        x: Input array.
    
    Returns:
        An array with applied softmax function
    """
    def compute_softmax(row):
        """ Compute softmax for a single array. """
        # Find max value for numerical stability
        row_max = max(row)
        # Calculate exponentials of adjusted values
        exp_values = [math.exp(item - row_max) for item in row]
        # Calculate the sum of the exponentials
        sum_exp_values = sum(exp_values)
        # Calculate softmax for each value in the row
        return [item / sum_exp_values for item in exp_values]

    # Check if x is a single list or a list of lists, and convert to list if necessary
    if not isinstance(x[0], list):
        x = [x]

    # Apply softmax to each row
    softmax_output = [compute_softmax(row) for row in x]

    # Flatten the output if the original input was a single list
    return softmax_output if len(softmax_output) > 1 else softmax_output[0]

def layer_norm(x, g, b, eps: float = 1e-5):
    """
    Applies layer normalization to the input array.
    
    Args:
        x: Input array.
        g: Scale array.
        b: Bias array.
        eps (float): Epsilon value for numerical stability.
    
    Returns:
        A layer normalized array.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(variance + eps) + b

def linear(x, w, b):
    """
    Performs a linear transformation on the input array.

    Args:
        x: Input array.
        w: Weight array.
        b: Bias array.
    
    Returns:
        numpy.ndarray: Linearly transformed array.
    """
    return x @ w + b

def ffn(x, c_fc, c_proj):
    """
    Applies feedforward neural network to the input array.

    Args:
        x: Input array.
        c_fc: Dictionary containing weights and biases for the feedforward layer.
        c_proj: Dictionary containing weights and biases for the projection layer.
    
    Returns:
        An array with applied feedforward neural network.
    """
    return linear(gelu(linear(x, **c_fc)), **c_proj)

def attention(q, k, v, mask):
    """
    Applies attention mechanism to the input arrays.

    Args:
        q: Query array.
        k: Key array.
        v: Value array.
        mask: Mask array
    
    Returns:
        An array with applied attention mechanism.
    """
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v

def mha(x, c_attn, c_proj, n_head):
    """
    Applies multi-head attention mechanism to the input array.

    Args:
        x: Input array.
        c_attn: Dictionary containing weights and biases for the attention layer.
        c_proj: Dictionary containing weights and biases for the projection layer.
        n_head: Number of attention heads.
    
    Returns:
        An array with applied multi-head attention mechanism.
    """
    x = linear(x, **c_attn)
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), np.split(x, 3, axis=-1)))
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]
    x = linear(np.hstack(out_heads), **c_proj)
    return x

def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    """
    A transformer block that applies multi-head attention and ffn to an input array.

    Args:
        x: The input array.
        mlp: A dictionary of ffn (feedforward neural network) parameters.
        attn: A dictionary of multi-head attention parameters.
        ln_1: A dictionary of LayerNorm parameters for the first LayerNorm layer.
        ln_2: A dictionary of LayerNorm paramters for the second LayerNorm layer.
        n_head: The number of heads for the multi-head attention.
    
    Returns:
        The output array of the transformer block.
    """
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x

def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    """
    A transformer network composed of several transformer blocks.

    Args:
        inputs: The input array.
        wte: The token embeddings array.
        wpe: The positional embeddings array.
        blocks: A list of dictionaries, each containing the parameters for a transformer block.
        ln_f: A dictionary of LayerNorm parameters for the final LayerNorm layer.
        n_head: The number of heads for multi-head attention.
    
    Returns:
        The output array of the transformer network.
    """
    x = wte[inputs] + wpe[range(len(inputs))]
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
    return layer_norm(x, **ln_f) @ wte.T

def generate(inputs, params, n_head, n_tokens_to_generate):
    """
    Generate new tokens given an initial sequence of tokens and a set of parameters.

    Args:
        inputs: The initial sequence of tokens as a list of integers.
        params: A dictionary containing the parameters of the GPT-2 model.
        n_head: The number of heads for multi-head attention.
        n_tokens_to_generate: The number of tokens to generate.
    
    Returns:
        The generates sequence of tokens as a list of integers.
    """
    from tqdm import tqdm
    for _ in tqdm(range(n_tokens_to_generate), "generating"):
        logits = gpt2(inputs, **params, n_head=n_head)
        next_id = np.argmax(logits[-1])
        inputs.append(int(next_id))
    return inputs[len(inputs) - n_tokens_to_generate :]

def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    input_ids = encoder.encode(prompt)
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    output_text = encoder.decode(output_ids)
    return output_text

if __name__ == "__main__":
    import fire
    fire.Fire(main)
