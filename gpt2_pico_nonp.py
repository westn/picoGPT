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
    def calculate_softmax_for_row(row):
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
    softmax_output = [calculate_softmax_for_row(row) for row in x]

    # Flatten the output if the original input was a single list
    return softmax_output if len(softmax_output) > 1 else softmax_output[0]

# layer_norm helper functions start
def calculate_mean(values):
    """Calculate the mean of a list of values."""
    return sum(values) / len(values)

def calculate_standard_deviation(values, mean, eps):
    """Calculate the standard deviation of a list of values based on the mean."""
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance + eps)

def normalize_values(values, mean, std_dev):
    """Normalize a list of values given the mean and standard deviation."""
    return [(value - mean) / std_dev for value in values]

def apply_scale_and_shift(normalized_values, scales, shifts):
    """
    Apply scaling and shifting to the normalized values.

    Args:
        normalized_values: List of normalized values.
        scales: List of scale factors.
        shifts: List of shift values.

    Returns:
        List of scaled and shifted values.
    """
    scaled_and_shifted_values = []
    for normalized, scale, shift in zip(normalized_values, scales, shifts):
        scaled = normalized * scale
        shifted = scaled + shift
        scaled_and_shifted_values.append(shifted)

    return scaled_and_shifted_values

# layer_norm helper functions end

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
    normalized_x = []
    for xi in x:
        mean_xi = calculate_mean(xi)
        std_dev_xi = calculate_standard_deviation(xi, mean_xi, eps)
        normalized_values = normalize_values(xi, mean_xi, std_dev_xi)
        scaled_and_shifted_values = apply_scale_and_shift(normalized_values, g, b)
        normalized_x.append(scaled_and_shifted_values)

    return normalized_x

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
