import torch

def compute_pca(tensor: torch.Tensor, C: int, height=37, width=37, input_channel=2048):
    """
    Applies PCA on the last dimension of the input tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape (B, 37, 37, 1024).
        C (int): The target feature dimension for PCA (C <= 1024).

    Returns:
        tuple: A tuple containing:
            - reduced_tensor (torch.Tensor): Tensor with shape (B, 37, 37, C).
            - mean (torch.Tensor): Mean vector used for centering, shape (1024,).
            - top_components (torch.Tensor): PCA projection matrix of shape (1024, C).
    """
    tensor = tensor.reshape(-1, height, width, input_channel)
    
    V, H, W, D = tensor.shape  # D should be 1024

    # Reshape to (B*H*W, 1024) so each row is a feature vector
    X = tensor.reshape(-1, D)

    # Compute mean along the feature dimension and center the data
    mean = X.mean(dim=0)
    X_centered = X - mean

    # Compute covariance matrix
    # Note: Divide by N - 1 for an unbiased estimate of covariance
    N = X_centered.shape[0]
    cov_matrix = torch.matmul(X_centered.T, X_centered) / (N - 1)

    # Since the covariance matrix is symmetric, we can use eigen-decomposition.
    # torch.linalg.eigh returns eigenvalues in ascending order.
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix.float())

    # Select the top C eigenvectors based on the eigenvalues.
    # We sort the eigenvalues in descending order and choose the corresponding eigenvectors.
    top_indices = torch.argsort(eigenvalues, descending=True)[:C]
    top_components = eigenvectors[:, top_indices]  # shape: (1024, C)

    # Project the centered data onto the top C eigenvectors
    X_reduced = torch.matmul(X_centered, top_components)  # shape: (B*H*W, C)
    
    # import pdb; pdb.set_trace()

    # Reshape the projected data back into (B, 37, 37, C)
    reduced_tensor = X_reduced.reshape(V, H, W, C)

    return reduced_tensor, mean, top_components