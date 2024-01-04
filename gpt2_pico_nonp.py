import numpy as np
from math import tanh, sqrt, pi, exp

# General helper functions start


def matmul(A, B):
    """
    Performs matrix multiplication on two matrices.

    Args:
        A: First matrix (list of lists).
        B: Second matrix (list of lists).

    Returns:
        List of lists: Result of matrix multiplication.
    """
    if not (isinstance(A, list) and isinstance(B, list)):
        raise TypeError("Both A and B must be lists")

    # Validate dimensions
    if len(A[0]) != len(B):
        raise ValueError("Incompatible dimensions for matrix multiplication")

    # Transpose of second matrix for easier row-column multiplication
    B_transposed = list(zip(*B))

    # Perform multiplication
    result = []
    for row in A:
        result_row = []
        for col in B_transposed:
            # Dot product of rows of A and columns of B
            dot_product = sum(a_elem * b_elem for a_elem, b_elem in zip(row, col))
            result_row.append(dot_product)
        result.append(result_row)

    return result


# General helper functions end


def gelu(x):
    """
    Applies Gaussian Error Linear Units activation function to the input array

    Args:
        x: Input array.

    Returns:
        An array with applied activation function.
    """

    def gelu_single_value(val: float) -> float:
        return 0.5 * val * (1 + tanh(sqrt(2 / pi) * (val + 0.044715 * val**3)))

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
        """Compute softmax for a single array."""
        # Find max value for numerical stability
        row_max = max(row)
        # Calculate exponentials of adjusted values
        exp_values = [exp(item - row_max) for item in row]
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
    return sqrt(variance + eps)


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
    # Perform matrix multiplication
    product = matmul(x, w)

    # Add bias to each element
    transformed = []
    for product_row, bias_element in zip(product, b):
        transformed_row = [element + bias_element for element in product_row]
        transformed.append(transformed_row)

    return transformed


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


# attention helper function start


def transpose(matrix):
    """
    Transposes a given 2D list (matrix).

    Args:
        matrix: A 2D list representing a matrix.

    Returns:
        A transposed 2D list (matrix).
    """
    return list(map(list, zip(*matrix)))


# attention helper function end


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
    qk_matmul = matmul(q, transpose(k))  # Replacing '@' operation
    scaled_attention_logits = [
        [ele / sqrt(len(q[0])) for ele in row] for row in qk_matmul
    ]

    # Adding mask
    for i in range(len(scaled_attention_logits)):
        for j in range(len(scaled_attention_logits[i])):
            scaled_attention_logits[i][j] += mask[i][j]

    attention_weights = softmax(scaled_attention_logits)
    output = matmul(attention_weights, v)
    return output


# mha helper functions start


def split_list(lst, n):
    """Splits a list into n approximately equal parts."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def get_qkv_heads(x, n_head):
    """
    Refactored part of the mha function to split x into qkv_heads without NumPy.

    Args:
        x: Input array (assumed to be a nested list).
        n_head: Number of attention heads.

    Returns:
        A list of qkv_heads.
    """
    # Split x into 3 equal parts along the last dimension (assuming x is a 2D list)
    split_x = split_list(x, 3)

    # Further split each part into n_head parts along the last dimension
    qkv_heads = [split_list(part, n_head) for part in split_x]

    # Flattening the list to match the structure of np.split().map().split()
    flattened_qkv_heads = []
    for group in zip(*qkv_heads):
        for item in group:
            flattened_qkv_heads.append(item)

    return flattened_qkv_heads


def create_lower_triangular_matrix(size):
    """
    Creates a lower triangular matrix filled with ones, with zeros elsewhere.

    Args:
        size (int): The size of the matrix (number of rows and columns).

    Returns:
        List[List[float]]: A lower triangular matrix.
    """
    return [[1.0 if j <= i else 0.0 for j in range(size)] for i in range(size)]


def apply_causal_mask(matrix, mask_value=-1e10):
    """
    Applies a causal mask to the given matrix by subtracting each element from 1
    and then multiplying by a large negative number (mask_value).

    Args:
        matrix (List[List[float]]): The matrix to apply the causal mask to.
        mask_value (float): The value to multiply after subtracting from 1.

    Returns:
        List[List[float]]: The matrix with the causal mask applied.
    """
    return [[(1 - element) * mask_value for element in row] for row in matrix]


def flatten_list(nested_list):
    """
    Flattens a nested list into a single list.

    Args:
        nested_list: A list of lists.

    Returns:
        A single flattened list.
    """
    return [item for sublist in nested_list for item in sublist]


# mha helper functions end


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
    qkv_heads = get_qkv_heads(x, n_head)
    matrix_size = len(x)  # Assuming x is a square 2D list
    lower_triangular_matrix = create_lower_triangular_matrix(matrix_size)
    causal_mask = apply_causal_mask(lower_triangular_matrix)
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]
    # Flatten the output heads into a single list
    flattened_out_heads = flatten_list(out_heads)
    # Apply the linear transformation
    x = linear(flattened_out_heads, **c_proj)
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
        The generated sequence of tokens as a list of integers.
    """
    from tqdm import tqdm

    def get_next_token_id(logits):
        """
        Find the index of the highest value in the last element of logits.

        Args:
            logits: A list of lists containing logits.

        Returns:
            An integer representing the index of the maximum logit value.
        """
        return max(range(len(logits)), key=lambda index: logits[index])

    for _ in tqdm(range(n_tokens_to_generate), "generating"):
        logits = gpt2(inputs, **params, n_head=n_head)
        next_id = get_next_token_id(logits[-1])
        inputs.append(int(next_id))
    return inputs[len(inputs) - n_tokens_to_generate :]


def main(
    prompt: str,
    n_tokens_to_generate: int = 40,
    model_size: str = "124M",
    models_dir: str = "models",
):
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
