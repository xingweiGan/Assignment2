import torch
import math

class FlashAttentionPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        FlashAttention-2 forward pass implementation in pure PyTorch
        
        Args:
            Q: Query tensor [batch_size, seq_len_q, d_model] or [batch_size, n_heads, seq_len_q, d_model]
            K: Key tensor [batch_size, seq_len_k, d_model] or [batch_size, n_heads, seq_len_k, d_model]
            V: Value tensor [batch_size, seq_len_k, d_model] or [batch_size, n_heads, seq_len_k, d_model]
            is_causal: Boolean flag for causal masking (ignored for now)
            
        Returns:
            O: Output tensor with same shape as Q
        """
        
        # Store original shape and flatten all leading dimensions
        original_shape = Q.shape
        
        # Get seq_len and d_model (always last two dimensions)
        seq_len_q, d_model = Q.shape[-2:]
        seq_len_k = K.shape[-2]
        
        # Flatten all leading dimensions into a single batch dimension
        Q = Q.view(-1, seq_len_q, d_model)
        K = K.view(-1, seq_len_k, d_model)
        V = V.view(-1, seq_len_k, d_model)
        
        # Now we always have shape [flattened_batch, seq_len, d_model]
        flattened_batch = Q.shape[0]
        
        # Choose tile sizes (must be at least 16x16)
        B_q = min(64, seq_len_q)  # Query tile size
        B_k = min(64, seq_len_k)  # Key tile size
        
        # Calculate number of tiles
        T_q = (seq_len_q + B_q - 1) // B_q
        T_k = (seq_len_k + B_k - 1) // B_k
        
        # Initialize output tensors
        O = torch.zeros_like(Q)
        L = torch.zeros(flattened_batch, seq_len_q, device=Q.device, dtype=Q.dtype)
        
        # Split tensors into tiles
        Q_tiles = Q.split(B_q, dim=1)
        K_tiles = K.split(B_k, dim=1) 
        V_tiles = V.split(B_k, dim=1)
        
        scale = 1.0 / math.sqrt(d_model)
        
        # Process each Q tile
        for i, Q_i in enumerate(Q_tiles):
            current_seq_len_q = Q_i.shape[1]
            
            # Initialize running statistics for this Q tile
            O_i = torch.zeros_like(Q_i)
            l_i = torch.zeros(flattened_batch, current_seq_len_q, device=Q.device, dtype=Q.dtype)
            m_i = torch.full((flattened_batch, current_seq_len_q), float('-inf'), device=Q.device, dtype=Q.dtype)
            
            # Process each K,V tile
            for j, (K_j, V_j) in enumerate(zip(K_tiles, V_tiles)):
                
                # Compute attention scores for current tile pair
                S_ij = torch.matmul(Q_i, K_j.transpose(-1, -2)) * scale  # [flattened_batch, B_q, B_k]
                
                # Compute new running maximum
                m_ij = torch.max(m_i.unsqueeze(-1), torch.max(S_ij, dim=-1, keepdim=True)[0]).squeeze(-1)
                
                # Compute rescaled probabilities for current tile
                P_tilde_ij = torch.exp(S_ij - m_ij.unsqueeze(-1))
                
                # Update running sum with rescaling
                exp_diff = torch.exp(m_i - m_ij)
                l_ij = exp_diff * l_i + torch.sum(P_tilde_ij, dim=-1)
                
                # Update output with rescaling of previous results
                O_i = torch.diag_embed(exp_diff).matmul(O_i) + torch.matmul(P_tilde_ij, V_j)
                
                # Update running statistics
                l_i = l_ij
                m_i = m_ij
            
            # Final normalization for this Q tile
            O_i = torch.diag_embed(1.0 / l_i).matmul(O_i)
            
            # Compute logsumexp for this tile
            L_i = m_i + torch.log(l_i)
            
            # Store results
            start_idx = i * B_q
            end_idx = start_idx + current_seq_len_q
            O[:, start_idx:end_idx, :] = O_i
            L[:, start_idx:end_idx] = L_i
        
        # Reshape back to original shape
        O = O.view(original_shape)
        L = L.view(original_shape[:-1])  # Remove last dimension (d_model) from L
        
        # Save for backward pass (including L)
        ctx.save_for_backward(Q, K, V, O, L)
        
        return O  # Return only O, not the tuple
    
    @staticmethod
    def backward(ctx, grad_O, grad_L):
        # For now, raise NotImplementedError as instructed
        raise NotImplementedError("Backward pass not implemented yet")

# Wrapper function to match the required interface
def flash_attention_pytorch(Q, K, V, is_causal=False):
    """
    Wrapper function for FlashAttention PyTorch implementation
    
    Args:
        Q, K, V: Input tensors
        is_causal: Causal masking flag
        
    Returns:
        O: Attention output
    """
    return FlashAttentionPyTorch.apply(Q, K, V, is_causal)