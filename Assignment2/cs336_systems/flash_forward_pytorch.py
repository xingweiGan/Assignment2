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
    def backward(ctx, dO):
        """
        FlashAttention-2 backward pass implementation
        Following equations (13)-(19) from the paper
        
        Args:
            ctx: Context object containing saved tensors from forward pass
            dO: Gradient flowing back from next layer [batch, seq_len, d_model]
            
        Returns:
            grad_Q, grad_K, grad_V, grad_is_causal: Gradients w.r.t. forward inputs
        """
        # Retrieve saved tensors from forward pass
        Q, K, V, O, L = ctx.saved_tensors
        original_shape = Q.shape
        seq_len_q, d_model = Q.shape[-2:]
        seq_len_k = K.shape[-2]
        # Get original shape and flatten (same as forward)
        Q = Q.view(-1, seq_len_q, d_model)
        K = K.view(-1, seq_len_k, d_model)
        V = V.view(-1, seq_len_k, d_model)
        O = O.view(-1, seq_len_q, d_model)
        L = L.view(-1, seq_len_q)
        dO = dO.view(-1, seq_len_q, d_model)
        
        flattened_batch = Q.shape[0]
        scale = 1.0 / math.sqrt(d_model)
        
        # Initialize gradient tensors
        grad_Q = torch.zeros_like(Q)
        grad_K = torch.zeros_like(K)
        grad_V = torch.zeros_like(V)
        
        # Use same tile sizes as forward pass
        B_q = min(64, seq_len_q)
        B_k = min(64, seq_len_k)
        
        # Split tensors into tiles
        Q_tiles = Q.split(B_q, dim=1)
        K_tiles = K.split(B_k, dim=1)
        V_tiles = V.split(B_k, dim=1)
        O_tiles = O.split(B_q, dim=1)
        L_tiles = L.split(B_q, dim=1)
        dO_tiles = dO.split(B_q, dim=1)
        
        # Compute D vector (equation after 19): D_i = rowsum(dO_i ⊙ O_i)
        D = torch.sum(dO * O, dim=-1)  # [flattened_batch, seq_len_q]
        D_tiles = D.split(B_q, dim=1)
        
        # Process each tile pair (i,j) where i=query tile, j=key/value tile
        for i, (Q_i, O_i, L_i, dO_i, D_i) in enumerate(zip(Q_tiles, O_tiles, L_tiles, dO_tiles, D_tiles)):
            current_seq_len_q = Q_i.shape[1]
            
            for j, (K_j, V_j) in enumerate(zip(K_tiles, V_tiles)):
                current_seq_len_k = K_j.shape[1]
                
                # Recompute attention scores and probabilities for this tile pair
                # Following equation (13): S = QK^T / √d
                S_ij = torch.matmul(Q_i, K_j.transpose(-1, -2)) * scale  # [batch, B_q, B_k]
                
                # Following equation (14): P_ij = exp(S_ij - L_i)
                # Note: L_i contains logsumexp, so this gives us normalized probabilities
                P_ij = torch.exp(S_ij - L_i.unsqueeze(-1))  # [batch, B_q, B_k]
                
                # Following equation (15): dV = P^T dO
                # Accumulate gradients for V tile j
                start_k = j * B_k
                end_k = start_k + current_seq_len_k
                grad_V[:, start_k:end_k, :] += torch.matmul(P_ij.transpose(-1, -2), dO_i)
                
                # Following equation (16): dP = dO V^T
                dP_ij = torch.matmul(dO_i, V_j.transpose(-1, -2))  # [batch, B_q, B_k]
                
                # Following equation (17): dS_ij = P_ij ⊙ (dP_ij - D_i)
                # D_i needs to be broadcasted to match dP_ij shape
                dS_ij = P_ij * (dP_ij - D_i.unsqueeze(-1))  # [batch, B_q, B_k]
                
                # Following equation (18): dQ = dS K / √d
                # Accumulate gradients for Q tile i
                start_q = i * B_q
                end_q = start_q + current_seq_len_q
                grad_Q[:, start_q:end_q, :] += torch.matmul(dS_ij, K_j) * scale
                
                # Following equation (19): dK = dS^T Q / √d
                # Accumulate gradients for K tile j
                start_k = j * B_k
                end_k = start_k + current_seq_len_k
                grad_K[:, start_k:end_k, :] += torch.matmul(dS_ij.transpose(-1, -2), Q_i) * scale
        
        # Reshape gradients back to original shape
        grad_Q = grad_Q.view(original_shape)
        grad_K = grad_K.view(original_shape)
        grad_V = grad_V.view(original_shape)
        
        return grad_Q, grad_K, grad_V, None  # None for is_causal gradient