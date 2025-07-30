import torch
import triton
import triton.language as tl
import math

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    # Calculate number of key tiles
    Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)
    
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    
    # Load Q_i from global memory - this is fixed for this program
    Q_i = tl.load(Q_block_ptr)
    Q_i = Q_i.to(tl.float32)  # Use float32 for precision
    
    # Initialize O_i, l_i, m_i (Algorithm 1 notation)
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)
    
    # Single loop iterating key tiles 1 ≤ j ≤ Tk
    for j in range(Tk):
        # Load K^(j), V^(j) from global memory
        K_j = tl.load(K_block_ptr)
        V_j = tl.load(V_block_ptr)
        K_j = K_j.to(tl.float32)
        V_j = V_j.to(tl.float32)
        
        # Compute tile of pre-softmax attention scores S_i^(j) = Q_i(K^(j))^T / √d
        S_ij = tl.dot(Q_i, tl.trans(K_j))
        S_ij = S_ij * scale
 
        # ← ← ← MODIFIED BELOW (was different implementation) ← ← ←
        # Apply causal masking if needed
        if is_causal:
            # Construct index vectors for queries and keys
            # Query indices for this tile
            q_start = query_tile_index * Q_TILE_SIZE
            q_indices = q_start + tl.arange(0, Q_TILE_SIZE)[:, None]  # Shape: (Q_TILE_SIZE, 1)
            
            # Key indices for current key tile j  
            k_start = j * K_TILE_SIZE
            k_indices = k_start + tl.arange(0, K_TILE_SIZE)[None, :]  # Shape: (1, K_TILE_SIZE)
            
            # Compare indices to form square mask of size Bq × Bk
            causal_mask = q_indices >= k_indices  # Shape: (Q_TILE_SIZE, K_TILE_SIZE)
            
            # For masked out elements, add -1e6 to attention scores
            S_ij = tl.where(causal_mask, S_ij, S_ij + (-1e6))
        # ← ← ← MODIFIED ABOVE ← ← ←




        # Compute m_i^(j) = max(m_i^(j-1), rowmax(S_i^(j)))
        m_i_new = tl.maximum(m_i, tl.max(S_ij, axis=1))
        
        # Compute P_i^(j) = exp(S_i^(j) - m_i^(j))
        P_ij = tl.exp(S_ij - m_i_new[:, None])
        
        # Compute l_i^(j) = exp(m_i^(j-1) - m_i^(j)) * l_i^(j-1) + rowsum(P_i^(j))
        l_i_new = tl.exp(m_i - m_i_new) * l_i + tl.sum(P_ij, axis=1)
        
        # Compute O_i^(j) = diag(exp(m_i^(j-1) - m_i^(j))) * O_i^(j-1) + P_i^(j) * V^(j)
        # First term: scale previous output
        scale_factor = tl.exp(m_i - m_i_new)
        O_i = O_i * scale_factor[:, None]
        
        # Second term: add new contribution
        # Cast P_ij to match V_j dtype before multiplication
        P_ij = P_ij.to(V_j.dtype)
        O_i = tl.dot(P_ij, V_j, acc=O_i)
        
        # Update running statistics
        m_i = m_i_new
        l_i = l_i_new
        
        # Advance block pointers at the end of the loop
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    
    # Final normalization: O_i = diag(l_i^(Tk))^(-1) * O_i^(Tk)
    O_i = O_i / l_i[:, None]
    
    # Compute L_i = m_i^(Tk) + log(l_i^(Tk))
    L_i = m_i + tl.log(l_i)
    
    # Cast O_i to appropriate dtype before writing to global memory
    output_dtype = O_block_ptr.type.element_ty
    O_i = O_i.to(output_dtype)
    
    # Write O_i to global memory as the i-th tile of O
    tl.store(O_block_ptr, O_i)
    
    # Write L_i to global memory as the i-th tile of L
    L_i = L_i.to(L_block_ptr.type.element_ty)
    tl.store(L_block_ptr, L_i)


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # Get dimensions using flexible approach
        seq_len_q, d_model = Q.shape[-2:]
        seq_len_k = K.shape[-2]
        
        # Calculate scale from model dimension
        scale = 1.0 / math.sqrt(d_model)
        
        # Store original shape for output reshaping
        original_shape = Q.shape
        
        # Tile sizes (can be tuned later)
        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64
        
        # Reshape tensors to flatten all batch dimensions
        Q = Q.view(-1, seq_len_q, d_model)
        K = K.view(-1, seq_len_k, d_model)
        V = V.view(-1, seq_len_k, d_model)
        
        # Get flattened batch size
        total_batch_size = Q.shape[0]
        
        # Create output tensors
        O = torch.empty_like(Q)
        L = torch.empty((total_batch_size, seq_len_q), dtype=torch.float32, device=Q.device)
        
        # Calculate grid dimensions
        Tq = triton.cdiv(seq_len_q, Q_TILE_SIZE)
        grid = (Tq, total_batch_size)
        # ← ← ← ADDED BELOW ← ← ←
        # Save the mask flag for backward
        ctx.is_causal = is_causal
        # ← ← ← ADDED ABOVE ← ← ←
        # Launch kernel
        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),  # stride_qb, stride_qq, stride_qd
            K.stride(0), K.stride(1), K.stride(2),  # stride_kb, stride_kk, stride_kd
            V.stride(0), V.stride(1), V.stride(2),  # stride_vb, stride_vk, stride_vd
            O.stride(0), O.stride(1), O.stride(2),  # stride_ob, stride_oq, stride_od
            L.stride(0), L.stride(1),               # stride_lb, stride_lq
            seq_len_q, seq_len_k,
            scale,
            D=d_model,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )
        
        # Reshape output back to original shape
        O = O.view(original_shape)
        ctx.save_for_backward(L)
        
        return O
    
    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass not implemented for this assignment
        raise NotImplementedError("Backward pass not implemented")